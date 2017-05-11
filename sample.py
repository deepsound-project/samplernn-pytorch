from model import SampleRNN, Predictor, Generator
from trainer import Trainer, sequence_nll_loss
from dataset import FolderDataset, DataLoader

import torch
from torch.utils.trainer import plugins

from librosa.output import write_wav

from time import time


def main():
    model = SampleRNN(
        frame_sizes=[16, 4], n_rnn=1, dim=1024, learn_h0=True, q_levels=256
    )
    predictor = Predictor(model).cuda()
    predictor.load_state_dict(torch.load('model.tar'))

    generator = Generator(predictor.model, cuda=True)

    t = time()
    samples = generator(5, 16000)
    print('generated in {}s'.format(time() - t))

    write_wav(
        'sample.wav',
        samples.cpu().float().numpy()[0, :],
        sr=16000,
        norm=True
    )

if __name__ == '__main__':
    main()
