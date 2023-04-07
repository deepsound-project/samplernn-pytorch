import matplotlib
matplotlib.use('Agg')

from model import Generator

import torch
from torch.autograd import Variable
from trainer.trainer_plugins.plugin import Plugin
from trainer.trainer_plugins.monitor import Monitor
from trainer.trainer_plugins.loss import LossMonitor

# from librosa.output import write_wav
import soundfile as sf
from matplotlib import pyplot
from librosa.util import normalize

from glob import glob
import os
import pickle
import time


class TrainingLossMonitor(LossMonitor):

    stat_name = 'training_loss'


class ValidationPlugin(Plugin):

    def __init__(self, val_dataset, test_dataset):
        super().__init__([(1, 'epoch')])
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def register(self, trainer):
        self.trainer = trainer
        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['log_epoch_fields'] = ['{last:.4f}']
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['log_epoch_fields'] = ['{last:.4f}']

    def epoch(self, idx):
        
        self.trainer.model.eval()

        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['last'] = self._evaluate(self.val_dataset)
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['last'] = self._evaluate(self.test_dataset)
        torch.cuda.empty_cache()
        self.trainer.model.train()

    def _evaluate(self, dataset):
        loss_sum = 0
        n_examples = 0
        for data in dataset:
            torch.cuda.empty_cache()
            batch_inputs = list(data[: -1])
            batch_target = data[-1]
            batch_size = batch_target.size()[0]
            
            batch_inputs[0] = batch_inputs[0].cuda() if self.trainer.cuda else batch_inputs[0]

            with torch.no_grad():
                batch_target = Variable(batch_target)
                if self.trainer.cuda:
                    batch_target = batch_target.cuda()


            batch_output = self.trainer.model(*batch_inputs)
            loss_sum += (self.trainer.criterion(batch_output, batch_target) * batch_size).data

            n_examples += batch_size

        return loss_sum / n_examples


class AbsoluteTimeMonitor(Monitor):

    stat_name = 'time'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 's')
        kwargs.setdefault('precision', 0)
        kwargs.setdefault('running_average', False)
        kwargs.setdefault('epoch_average', False)
        super(AbsoluteTimeMonitor, self).__init__(*args, **kwargs)
        self.start_time = None

    def _get_value(self, *args):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time


class SaverPlugin(Plugin):

    last_pattern = 'ep{}-it{}'
    best_pattern = 'best-ep{}-it{}'

    def __init__(self, checkpoints_path, keep_old_checkpoints):
        super().__init__([(1, 'epoch')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
        torch.save(
            self.trainer.model.state_dict(),
            os.path.join(
                self.checkpoints_path,
                self.last_pattern.format(epoch_index, self.trainer.iterations)
            )
        )

        cur_val_loss = self.trainer.stats['validation_loss']['last']
        if cur_val_loss < self._best_val_loss:
            self._clear(self.best_pattern.format('*', '*'))
            torch.save(
                self.trainer.model.state_dict(),
                os.path.join(
                    self.checkpoints_path,
                    self.best_pattern.format(
                        epoch_index, self.trainer.iterations
                    )
                )
            )
            self._best_val_loss = cur_val_loss

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)


class GeneratorPlugin(Plugin):

    pattern = 'ep{}-s{}.wav'

    def __init__(self, samples_path, n_samples, sample_length, sample_rate):
        super().__init__([(1, 'epoch')])
        self.samples_path = samples_path
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.sample_rate = sample_rate

    def register(self, trainer):
        self.generate = Generator(trainer.model.model, trainer.cuda)

    def epoch(self, epoch_index):
        samples = self.generate(self.n_samples, self.sample_length) \
                      .cpu().float().numpy()
        for i in range(self.n_samples):
            sample_norm = normalize(samples[i, :], axis=0)
            sf.write(
                os.path.join(
                    self.samples_path, self.pattern.format(epoch_index, i + 1)
                ),
                sample_norm, self.sample_rate
            )



class StatsPlugin(Plugin):

    data_file_name = 'stats.pkl'
    plot_pattern = '{}.svg'

    def __init__(self, results_path, iteration_fields, epoch_fields, plots):
        super().__init__([(1, 'iteration'), (1, 'epoch')])
        self.results_path = results_path

        self.iteration_fields = self._fields_to_pairs(iteration_fields)
        self.epoch_fields = self._fields_to_pairs(epoch_fields)
        self.plots = plots
        self.data = {
            'iterations': {
                field: []
                for field in self.iteration_fields + [('iteration', 'last')]
            },
            'epochs': {
                field: []
                for field in self.epoch_fields + [('iteration', 'last')]
            }
        }

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, *args):
        for (field, stat) in self.iteration_fields:
            self.data['iterations'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['iterations']['iteration', 'last'].append(
            self.trainer.iterations
        )

    def epoch(self, epoch_index):
        for (field, stat) in self.epoch_fields:
            self.data['epochs'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['epochs']['iteration', 'last'].append(
            self.trainer.iterations
        )

        data_file_path = os.path.join(self.results_path, self.data_file_name)
        with open(data_file_path, 'wb') as f:
            pickle.dump(self.data, f)

        for (name, info) in self.plots.items():
            x_field = self._field_to_pair(info['x'])

            try:
                y_fields = info['ys']
            except KeyError:
                y_fields = [info['y']]

            labels = list(map(
                lambda x: ' '.join(x) if type(x) is tuple else x,
                y_fields
            ))
            y_fields = self._fields_to_pairs(y_fields)

            try:
                formats = info['formats']
            except KeyError:
                formats = [''] * len(y_fields)

            pyplot.gcf().clear()

            for (y_field, format, label) in zip(y_fields, formats, labels):
                if y_field in self.iteration_fields:
                    part_name = 'iterations'
                else:
                    part_name = 'epochs'

                xs = self._to_cpu(self.data[part_name][x_field])
                ys = self._to_cpu(self.data[part_name][y_field])
                
                pyplot.plot(xs, ys, format, label=label)

            if 'log_y' in info and info['log_y']:
                pyplot.yscale('log')

            pyplot.legend()
            pyplot.savefig(
                os.path.join(self.results_path, self.plot_pattern.format(name))
            )

    def _to_cpu(self, xs):
        xs_ = list()
        for element in xs:
            if type(element) == torch.Tensor:
                xs_.append(element.cpu())
            else:
                xs_.append(element)
        return xs_

    @staticmethod
    def _field_to_pair(field):
        if type(field) is tuple:
            return field
        else:
            return (field, 'last')

    @classmethod
    def _fields_to_pairs(cls, fields):
        return list(map(cls._field_to_pair, fields))


class CometPlugin(Plugin):

    def __init__(self, experiment, fields):
        super().__init__([(1, 'epoch')])

        self.experiment = experiment
        self.fields = [
            field if type(field) is tuple else (field, 'last')
            for field in fields
        ]

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        for (field, stat) in self.fields:
            self.experiment.log_metric(field, self.trainer.stats[field][stat])
        self.experiment.log_epoch_end(epoch_index)
