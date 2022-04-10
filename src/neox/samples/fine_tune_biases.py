"""
---
title: Fine Tune GPT-NeoX
summary: >
    Fine tune GPT-NeoX biases with Fairscale pipeline parallel module
---

# Fine Tune GPT-NeoX

This shows how to fine tune GPT-NeoX with pipeline parallelism.
"""

# Imports
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

# List of layers to load. This is used for testing.
# You can assign a subset of layers like `{0, 1}` so that it only loads
# the first to transformer layers.
LAYERS = None


def main():
    """
    ## Train GPT-NeoX
    """

    # Create the experiment for tracking
    experiment.create(name='finetune_neox_biases', comment='Pipeline parallel', writers={'screen', 'web_api'})

    # Load layers
    layers = load_layers(LAYERS)

    # Mark `requires_grad=True` for biases using a [helper function](../utils/training.html).
    train_biases_only(layers)

    # Create the pipeline parallel model
    with monit.section('Pipe'):
        # Number of GPUs
        n_gpus = min(16, torch.cuda.device_count())
        # [Get the distribution of layers across the GPUs](../utils/__init__.py)
        balance = balance_layers(len(layers), n_gpus)
        # Get the GPU references
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
        # Create the pipeline parallel model
        pipe_model = fairscale.nn.Pipe(nn.Sequential(*layers),
                                       balance=balance,
                                       devices=devices,
                                       chunks=8)

    # Load [dataset](../dataset.html)
    dataset = get_training_data(1024)

    # Create data loader
    train_dl = DataLoader(dataset,
                          batch_size=8,
                          sampler=RandomSampler(dataset, replacement=True))

    # Initialize optimizer
    optimizer = optim.Adam(get_trainable_params(pipe_model), lr=1e-6)

    # Train the model using the [helper function](../utils/training.html)
    with experiment.start():
        for epoch in monit.loop(16):
            train(pipe_model, optimizer, train_dl, torch.device('cuda:0'), 10)
            tracker.new_line()


#
if __name__ == '__main__':
    main()
