"""
---
title: Training Utilities and Helpers
summary: >
    Utilities and helper functions for model training
---

# Training Utilities and Helpers
"""
from typing import List

import torch.nn as nn
import torch.utils.data
import torch.optim
from torch.cuda import amp

from labml import monit, tracker
from labml.logger import inspect


def get_trainable_params(model: nn.Module):
    """
    ### Get trainable parameters

    :param model: is the model to train
    :return: a list of parameters for training
    """

    # Get all parameters
    params = list(model.parameters())
    # Filter parameters that require gradients
    trainable_params = [p for p in params if p.requires_grad]
    # Log
    inspect(params=len(params), params_training=len(trainable_params))

    #
    return trainable_params


def train_biases_only(layers: List[nn.Module]):
    """
    ### Train only biases

    This sets `requires_grad` to `False` in all parameters except biases.
    We use this for fine-tuning, when it's too slow/expensive to train all parameters.

    :param layers: is the list of layers
    """

    for layer in layers:
        # Set `requires_grad` to `False` for the entire layer.
        layer.requires_grad_(False)
        #
        for n, p in layer.named_parameters():
            # Set `requires_grad` to `True` only for biases
            if 'bias' in n:
                p.requires_grad_(True)


def train(model: nn.Module, optimizer: torch.optim.Adam,
          train_loader: torch.utils.data.DataLoader,
          device: torch.device, train_log_interval: int):
    """
    ## Simple trainer

    This trains the `model` for a single epoch.

    :param model: is the model
    :param optimizer: is the optimizer
    :param train_loader: is the training data loader
    :param device: is the device for inputs
    :param train_log_interval:  is the logging frequency
    """

    # Set model for train
    model.train()

    # Cross-entropy loss
    loss_func = nn.CrossEntropyLoss()

    # Iterate through the batches
    for batch_idx, (data, target) in monit.enum('Train', train_loader):
        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass
        with monit.section('Forward pass'):
            output = model(data.to(device))
        # Move targets to the same device as output
        target = target.to(output.device)
        # Calculate loss
        loss = loss_func(output.view(target.numel(), -1), target.view(-1))

        # Get predictions
        pred = output.argmax(dim=-1)
        # Calculate accuracy
        accuracy = pred.eq(target).sum().item() / pred.numel()

        # Backward pass
        with monit.section('Backward pass'):
            loss.backward()

        # Optimize
        with monit.section('Optimize'):
            optimizer.step()

        # Log the stats
        tracker.add_global_step()
        tracker.save({'loss.train': loss, 'acc.train': accuracy * 100})

        # Log model stats like gradients and weights once in a while
        # if batch_idx % train_log_interval == 0:
        #     tracker.save(model=model)

    # Log model stats like gradients and weights at the end of the epoch
    # tracker.save(model=model)


def train_amp(model: nn.Module, optimizer: torch.optim.Adam,
              train_loader: torch.utils.data.DataLoader,
              device: torch.device, scaler: amp.GradScaler,
              train_log_interval: int):
    """
    ## Simple trainer

    This trains the `model` for a single epoch.

    :param model: is the model
    :param optimizer: is the optimizer
    :param train_loader: is the training data loader
    :param device: is the device for inputs
    :param train_log_interval:  is the logging frequency
    """

    # Set model for train
    model.train()

    # Cross-entropy loss
    loss_func = nn.CrossEntropyLoss()

    # Iterate through the batches
    for batch_idx, (data, target) in monit.enum('Train', train_loader):
        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass
        with amp.autocast():
            with monit.section('Forward pass'):
                output = model(data.to(device))
            # Move targets to the same device as output
            target = target.to(output.device)

            # Calculate loss
            loss = loss_func(output.view(target.numel(), -1), target.view(-1))

        tracker.add({'loss.unscaled': loss})
        # Get predictions
        pred = output.argmax(dim=-1)
        # Calculate accuracy
        accuracy = pred.eq(target).sum().item() / pred.numel()

        # Backward pass
        loss = scaler.scale(loss)

        with monit.section('Backward pass'):
            loss.backward()

        # Optimize
        with monit.section('Optimize'):
            optimizer.step()

        # Log the stats
        tracker.add_global_step()
        tracker.save({'loss.train': loss, 'acc.train': accuracy * 100})

        # Log model stats like gradients and weights once in a while
        # if batch_idx % train_log_interval == 0:
        #     tracker.save(model=model)

    # Log model stats like gradients and weights at the end of the epoch
    # tracker.save(model=model)
