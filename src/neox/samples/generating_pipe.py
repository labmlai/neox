"""
---
title: Generate Text with GPT-NeoX using Pipeline Parallelism
summary: >
     Generate Text with GPT-NeoX using Fairscale Pipeline Parallelism
---

#  Generate Text with GPT-NeoX using Pipeline Parallelism

This shows how to generate text from GPT-NeoX with pipeline parallelism.
"""

# Imports
from typing import List

import fairscale
import torch
from torch import nn

from labml import monit
from neox.utils import load_layers, get_tokens, print_tokens, balance_layers
from neox.utils.cache import get_cache

# List of layers to load. This is used for testing.
# You can assign a subset of layers like `{0, 1}` so that it only loads
# the first to transformer layers.
LAYERS = None

# Prompt to complete
PROMPT = 'Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German'


def infer(model: nn.Module, ids: List[int], device: torch.device):
    """
    ### Predict the next token

    :param model: is the model
    :param ids: are the input token ids
    :param device: is the device of the model
    """

    # Call the model
    with torch.no_grad():
        x = torch.tensor(ids)[None, :].to(device)
        x = model(x)

    # Return the outputs
    return x[0].max(dim=-1)[1].tolist()


def generate():
    """
    ## Generate text
    """

    # Setup [cache](../utils/cache.html) to cache intermediate key/value pairs for faster generation
    cache = get_cache()
    cache.set('use_cache', True)

    # Load layers
    layers = load_layers(LAYERS)

    # Create pipeline parallel model
    with monit.section('Pipe'):
        # Number of GPUs
        n_gpus = min(4, torch.cuda.device_count())
        # [Get the distribution of layers across the GPUs](../utils/__init__.py)
        balance = balance_layers(len(layers), n_gpus)
        # Get the GPU references
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
        # Create the pipeline parallel model
        pipe_model = fairscale.nn.Pipe(nn.Sequential(*layers),
                                       balance=balance,
                                       devices=devices)

    # Get token ids
    ids = get_tokens(PROMPT)

    # Run the model
    cache.set('state_ids', (None, 1))
    next_token = infer(pipe_model, ids, pipe_model.devices[0])[-1]

    # Append the predicted token
    ids += [next_token]

    # Predict 100 tokens
    for i in range(1, 100):
        # Set the state to use cached activations
        cache.set('state_ids', (i, i + 1))
        # Get next token. Note that we only feed the last token to the model because
        # we cache the key/value pairs of previous tokens.
        next_token = infer(pipe_model, [next_token], pipe_model.devices[0])[-1]
        # Append the predicted token
        ids += [next_token]
        # Print
        print_tokens(ids, [ids])


#
if __name__ == '__main__':
    generate()
