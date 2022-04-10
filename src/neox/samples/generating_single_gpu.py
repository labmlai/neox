"""
---
title: Generate Text with GPT-NeoX by Evaluating Layer by Layer
summary: >
     Generate Text with GPT-NeoX by evaluating layer by layer
---

#  Generate Text with GPT-NeoX by Evaluating Layer by Layer

This shows how to generate text from GPT-NeoX with a single GPU.
"""

# Imports
from typing import List

import torch
from torch import nn

from labml import monit
from neox.utils import load_layers, get_tokens, print_tokens
from neox.utils.cache import get_cache

# List of layers to load. This is used for testing.
# You can assign a subset of layers like `{0, 1}` so that it only loads
# the first to transformer layers.
LAYERS = None

# Prompt to complete
PROMPT = 'Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German'


def infer(layers: List[nn.Module], ids: List[int], device: torch.device):
    """
    ### Predict the next token

    :param layers: is the list of layers
    :param ids: are the input token ids
    :param device: is the device of the model
    """

    # Offload to CPU
    offload = torch.device('cpu')
    # CUDA stream for async loading and de-loading. This is still WIP.
    s = torch.cuda.Stream()
    #
    with torch.no_grad():
        # Get the tokens
        x = torch.tensor(ids)[None, :].to(device)
        # Iterate through the layers
        for layer in layers:
            # Move the layer to device. Should pre-load.
            layer.to(device)
            # Evaluate the layer
            x = layer(x)
            # Offload (async)
            with torch.cuda.stream(s):
                layer.to(offload)

    # Return predicted token
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

    # Device
    device = torch.device('cuda:0')

    # Get token ids
    ids = get_tokens(PROMPT)

    # Run the model
    cache.set('state_ids', (None, 1))
    with monit.section('Infer'):
        next_token = infer(layers, ids, device)[-1]

    # Append the predicted token
    ids += [next_token]

    # Predict 100 tokens
    for i in range(1, 100):
        # Set the state to use cached activations
        cache.set('state_ids', (i, i + 1))
        # Get next token. Note that we only feed the last token to the model because
        # we cache the key/value pairs of previous tokens.
        with monit.section('Infer'):
            next_token = infer(layers, [next_token], device)[-1]
        # Append the predicted token
        ids += [next_token]
        # Print
        print_tokens(ids, [ids])


#
if __name__ == '__main__':
    generate()
