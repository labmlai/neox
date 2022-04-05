"""
---
title: GPT-NeoX Tokenizer
summary: >
    Loads the GPT-NeoX tokenizer
---

# GPT-NeoX Tokenizer

This initializes a Hugging Face tokenizer from the downloaded vocabulary.
"""

from tokenizers import Tokenizer

from labml import lab
from labml.logger import inspect


def get_tokenizer() -> Tokenizer:
    """
    ### Load NeoX Tokenizer

    :return: the tokenizer
    """
    vocab_file = lab.get_data_path() / 'neox' / 'slim_weights' / '20B_tokenizer.json'
    tokenizer = Tokenizer.from_file(str(vocab_file))

    return tokenizer


def _test():
    """
    #### Testing code
    """
    tokenizer = get_tokenizer()

    batch = tokenizer.encode_batch(['Hello how are you', 'My name is Varuna'])

    inspect(batch, _expand=True, _n=-1)


#
if __name__ == '__main__':
    _test()
