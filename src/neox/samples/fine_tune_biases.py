import fairscale
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data
from torch.utils.data import DataLoader, RandomSampler

from labml import tracker, experiment, monit
from neox.data import get_training_data
from neox.utils import load_layers, balance_layers
from neox.utils.training import train, get_trainable_params, train_biases_only

LAYERS = None


def main():
    experiment.create(name='finetune_neox_biases', comment='Pipeline parallel', writers={'screen', 'web_api'})

    layers = load_layers(LAYERS)

    train_biases_only(layers)

    with monit.section('Pipe'):
        n_gpus = min(16, torch.cuda.device_count())
        balance = balance_layers(len(layers), n_gpus)
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
        pipe_model = fairscale.nn.Pipe(nn.Sequential(*layers),
                                       balance=balance,
                                       devices=devices,
                                       chunks=8)

    dataset = get_training_data(1024)

    train_dl = DataLoader(dataset,
                          batch_size=8,
                          sampler=RandomSampler(dataset, replacement=True))

    optimizer = optim.Adam(get_trainable_params(pipe_model), lr=1e-6)

    with experiment.start():
        for epoch in monit.loop(16):
            train(pipe_model, optimizer, train_dl, torch.device('cuda:0'), 10)
            tracker.new_line()


if __name__ == '__main__':
    main()
