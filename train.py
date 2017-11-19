# CometML needs to be imported first.
try:
    import comet_ml
except ImportError:
    pass

from model import SampleRNN, Predictor
from optim import gradient_clipping
from nn import sequence_nll_loss_bits
from trainer import Trainer
from trainer.plugins import (
    TrainingLossMonitor, ValidationPlugin, AbsoluteTimeMonitor, SaverPlugin,
    GeneratorPlugin, StatsPlugin
)
from dataset import FolderDataset, DataLoader

import torch
from torch.utils.trainer.plugins import Logger

from natsort import natsorted

from functools import reduce
import os
import shutil
import sys
from glob import glob
import re
import argparse


default_params = {
    # model parameters
    'n_rnn': 1,
    'dim': 1024,
    'learn_h0': True,
    'q_levels': 256,
    'seq_len': 1024,
    'weight_norm': True,
    'batch_size': 128,
    'val_frac': 0.1,
    'test_frac': 0.1,

    # training parameters
    'keep_old_checkpoints': False,
    'datasets_path': 'datasets',
    'results_path': 'results',
    'epoch_limit': 1000,
    'resume': True,
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'loss_smoothing': 0.99,
    'cuda': True,
    'comet_key': None
}

tag_params = [
    'exp', 'frame_sizes', 'n_rnn', 'dim', 'learn_h0', 'q_levels', 'seq_len',
    'batch_size', 'dataset', 'val_frac', 'test_frac'
]

def param_to_string(value):
    if isinstance(value, bool):
        return 'T' if value else 'F'
    elif isinstance(value, list):
        return ','.join(map(param_to_string, value))
    else:
        return str(value)

def make_tag(params):
    return '-'.join(
        key + ':' + param_to_string(params[key])
        for key in tag_params
        if key not in default_params or params[key] != default_params[key]
    )

def setup_results_dir(params):
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    tag = make_tag(params)
    results_path = os.path.abspath(params['results_path'])
    ensure_dir_exists(results_path)
    results_path = os.path.join(results_path, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    elif not params['resume']:
        shutil.rmtree(results_path)
        os.makedirs(results_path)

    for subdir in ['checkpoints', 'samples']:
        ensure_dir_exists(os.path.join(results_path, subdir))

    return results_path

def load_last_checkpoint(checkpoints_path):
    checkpoints_pattern = os.path.join(
        checkpoints_path, SaverPlugin.last_pattern.format('*', '*')
    )
    checkpoint_paths = natsorted(glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            SaverPlugin.last_pattern.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
    else:
        return None

def tee_stdout(log_path):
    log_file = open(log_path, 'a', 1)
    stdout = sys.stdout

    class Tee:

        def write(self, string):
            log_file.write(string)
            stdout.write(string)

        def flush(self):
            log_file.flush()
            stdout.flush()

    sys.stdout = Tee()

def make_data_loader(overlap_len, params):
    path = os.path.join(params['datasets_path'], params['dataset'])
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, overlap_len, params['q_levels'], split_from, split_to
        )
        return DataLoader(
            dataset,
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            overlap_len=overlap_len,
            shuffle=(not eval),
            drop_last=(not eval)
        )
    return data_loader

def init_comet(params, trainer):
    if params['comet_key'] is not None:
        from comet_ml import Experiment
        from trainer.plugins import CometPlugin
        experiment = Experiment(api_key=params['comet_key'], log_code=False)
        hyperparams = {
            name: param_to_string(params[name]) for name in tag_params
        }
        experiment.log_multiple_params(hyperparams)
        trainer.register_plugin(CometPlugin(
            experiment, [
                ('training_loss', 'epoch_mean'),
                'validation_loss',
                'test_loss'
            ]
        ))

def main(exp, frame_sizes, dataset, **params):
    params = dict(
        default_params,
        exp=exp, frame_sizes=frame_sizes, dataset=dataset,
        **params
    )

    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    model = SampleRNN(
        frame_sizes=params['frame_sizes'],
        n_rnn=params['n_rnn'],
        dim=params['dim'],
        learn_h0=params['learn_h0'],
        q_levels=params['q_levels'],
        weight_norm=params['weight_norm']
    )
    predictor = Predictor(model)
    if params['cuda']:
        model = model.cuda()
        predictor = predictor.cuda()

    optimizer = gradient_clipping(torch.optim.Adam(predictor.parameters()))

    data_loader = make_data_loader(model.lookback, params)
    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']

    trainer = Trainer(
        predictor, sequence_nll_loss_bits, optimizer,
        data_loader(0, val_split, eval=False),
        cuda=params['cuda']
    )

    checkpoints_path = os.path.join(results_path, 'checkpoints')
    checkpoint_data = load_last_checkpoint(checkpoints_path)
    if checkpoint_data is not None:
        (state_dict, epoch, iteration) = checkpoint_data
        trainer.epochs = epoch
        trainer.iterations = iteration
        predictor.load_state_dict(state_dict)

    trainer.register_plugin(TrainingLossMonitor(
        smoothing=params['loss_smoothing']
    ))
    trainer.register_plugin(ValidationPlugin(
        data_loader(val_split, test_split, eval=True),
        data_loader(test_split, 1, eval=True)
    ))
    trainer.register_plugin(AbsoluteTimeMonitor())
    trainer.register_plugin(SaverPlugin(
        checkpoints_path, params['keep_old_checkpoints']
    ))
    trainer.register_plugin(GeneratorPlugin(
        os.path.join(results_path, 'samples'), params['n_samples'],
        params['sample_length'], params['sample_rate']
    ))
    trainer.register_plugin(
        Logger([
            'training_loss',
            'validation_loss',
            'test_loss',
            'time'
        ])
    )
    trainer.register_plugin(StatsPlugin(
        results_path,
        iteration_fields=[
            'training_loss',
            ('training_loss', 'running_avg'),
            'time'
        ],
        epoch_fields=[
            'validation_loss',
            'test_loss',
            'time'
        ],
        plots={
            'loss': {
                'x': 'iteration',
                'ys': [
                    'training_loss',
                    ('training_loss', 'running_avg'),
                    'validation_loss',
                    'test_loss',
                ],
                'log_y': True
            }
        }
    ))

    init_comet(params, trainer)

    trainer.run(params['epoch_limit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()

    parser.add_argument('--exp', required=True, help='experiment name')
    parser.add_argument(
        '--frame_sizes', nargs='+', type=int, required=True,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='dataset name - name of a directory in the datasets path \
              (settable by --datasets_path)'
    )
    parser.add_argument(
        '--n_rnn', type=int, help='number of RNN layers in each tier'
    )
    parser.add_argument(
        '--dim', type=int, help='number of neurons in every RNN and MLP layer'
    )
    parser.add_argument(
        '--learn_h0', type=parse_bool,
        help='whether to learn the initial states of RNNs'
    )
    parser.add_argument(
        '--q_levels', type=int,
        help='number of bins in quantization of audio samples'
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='how many samples to include in each truncated BPTT pass'
    )
    parser.add_argument(
        '--weight_norm', type=parse_bool,
        help='whether to use weight normalization'
    )
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument(
        '--val_frac', type=float,
        help='fraction of data to go into the validation set'
    )
    parser.add_argument(
        '--test_frac', type=float,
        help='fraction of data to go into the test set'
    )
    parser.add_argument(
        '--keep_old_checkpoints', type=parse_bool,
        help='whether to keep checkpoints from past epochs'
    )
    parser.add_argument(
        '--datasets_path', help='path to the directory containing datasets'
    )
    parser.add_argument(
        '--results_path', help='path to the directory to save the results to'
    )
    parser.add_argument('--epoch_limit', help='how many epochs to run')
    parser.add_argument(
        '--resume', type=parse_bool, default=True,
        help='whether to resume training from the last checkpoint'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the training data and generated sound'
    )
    parser.add_argument(
        '--n_samples', type=int,
        help='number of samples to generate in each epoch'
    )
    parser.add_argument(
        '--sample_length', type=int,
        help='length of each generated sample (in samples)'
    )
    parser.add_argument(
        '--loss_smoothing', type=float,
        help='smoothing parameter of the exponential moving average over \
              training loss, used in the log and in the loss plot'
    )
    parser.add_argument(
        '--cuda', type=parse_bool,
        help='whether to use CUDA'
    )
    parser.add_argument(
        '--comet_key', help='comet.ml API key'
    )

    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
