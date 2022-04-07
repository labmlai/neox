"""
---
title: Utilities and Helpers
summary: >
    Utilities and helper functions
---

# Utilities and Helpers

* [Cache for intermediate activations (for faster inference)](cache.html)
* [Utilities for training and fine-tuning](training.html)
"""

from pathlib import Path
from typing import List, Optional, Set

import torch
from tokenizers import Tokenizer

from labml import logger, monit
from labml.logger import inspect, Text
from neox.checkpoint import get_checkpoint_files, load_checkpoint_files
from neox.model import get_layers
from neox.tokenizer import get_tokenizer

# Tokenizer singleton
_TOKENIZER: Optional[Tokenizer] = None

# Sample texts for testing
with open(str(Path(__file__).parent / 'sample.txt'), 'r') as f:
    SAMPLE_LONG_TEXT = f.read()

SAMPLE_SHORT_TEXT = 'Good morning. How are you doing today?'


def get_tokens(text: str) -> List[int]:
    """
    ### Get token ids

    :param text: is the text to tokenize
    :return: the token ids
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()
    return _TOKENIZER.encode_batch([text])[0].ids


def get_sample_tokens(is_long: bool = False) -> List[int]:
    """
    ### Get sample

    :param is_long: is whether to get the short or long sample text
    :return: the token ids of the sample text
    """
    text = SAMPLE_LONG_TEXT if is_long else SAMPLE_SHORT_TEXT
    return get_tokens(text)


def print_token_outputs(ids: List[int], *xs: torch.Tensor):
    """
    ### Print tokens from model outputs

    Pretty prints target tokens along side outputs from the model(s).

    :param ids: are the target token ids
    :param xs: are the model(s) outputs
    """
    ids = ids + [-1]
    xs = [[-1] + x[0].max(dim=-1)[1].tolist() for x in xs]

    print_tokens(ids, xs)


def print_tokens(target: List[int], others: List[List[int]]):
    """
    ### Print tokens

    Pretty prints tokens for comparison

    :param target: are the target token ids
    :param others: are the sampled outputs from the model(s)
    """

    # Load tokenizer
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()

    # Convert the tokens to list of strings
    text = []
    for i in range(len(target)):
        tokens = [_TOKENIZER.decode([target[i]]) if target[i] != -1 else '---']
        for j in range(len(others)):
            tokens.append(_TOKENIZER.decode([others[j][i]]) if others[j][i] != -1 else '---')

        text.append(tokens)

    # Stats
    correct = [0 for _ in others]
    total = 0

    # Iterate through tokens
    for i in range(len(target)):
        parts = [(f'{i}: ', Text.meta)]
        parts += [('"', Text.subtle), (text[i][0], Text.subtle), ('"', Text.subtle), '\t']

        # Empty target
        if target[i] == -1:
            for j in range(len(others)):
                parts += [('"', Text.subtle), (text[i][j + 1], Text.subtle), ('"', Text.subtle), '\t']

            logger.log(parts)
            continue

        # Number of tokens
        total += 1

        # Other outputs
        for j in range(len(others)):
            correct[j] += 1 if others[j][i] == target[i] else 0

            parts += [('"', Text.subtle),
                      (text[i][j + 1], Text.success if others[j][i] == target[i] else Text.danger),
                      ('"', Text.subtle), '\t']

        logger.log(parts)

    # Stats
    parts = [(f'{total}', Text.highlight), '\t']
    for j in range(len(others)):
        parts += [(f'{correct[j]}', Text.value), '\t']
    logger.log(parts)


def _test_sample_tokens():
    """
    Test sample tokens
    """
    ids = get_sample_tokens(True)
    inspect(ids)

    text = [(t, _TOKENIZER.decode([t])) for t in ids]
    inspect(text)

    inspect(_TOKENIZER.decode(ids))


def load_layers(filter_layers: Optional[Set[int]]):
    """
    ### Load GPT-NeoX layers

    This is a helper function to initialize andn load the layers.

    :param filter_layers: are the layers to be filters. If `None` all layers will be loaded.
    :return: the list of loaded layers
    """
    with torch.no_grad():
        layers = []
        with monit.section("Layers"):
            for i, (layer, files) in enumerate(zip(get_layers(filter_layers=filter_layers), get_checkpoint_files())):
                if layer is None or files is None:
                    continue
                layer.load_state(*load_checkpoint_files(files))

                layers.append(layer)

                monit.progress(i / 49)

    return layers


def balance_layers(n_layers: int, n_chunks: int):
    """
    ### Balance layers

    Split the `n_layers` into `n_chunks`. This is used for pipeline parallel training.

    :param n_layers: is the number of layers
    :param n_chunks: is the number of chunks
    :return: returns a list with the number of layers for each chunk
    """
    balance = []
    for i in range(n_chunks):
        balance.append((n_layers - sum(balance)) // (n_chunks - i))

    return reversed(balance)


def _test_balance():
    """
    Test balancing
    """
    inspect(balance_layers(45, 4))


#
if __name__ == '__main__':
    _test_sample_tokens()
