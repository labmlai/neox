"""
---
title: GPT-NeoX Checkpoints
summary: >
    Code to download checkpoints and helpers to load them.
---

# GPT-NeoX Checkpoints

"""
from typing import Dict, Union, Tuple

import torch
from torch import nn

from labml import monit, lab, logger
from labml.logger import Text
from labml.utils.download import download_file

# Parent url
CHECKPOINTS_URL = 'https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/'
# Download path
CHECKPOINTS_DOWNLOAD_PATH = lab.get_data_path() / 'neox' / 'slim_weights'


def get_files_to_download():
    """
    ### Get files to download

    :return: a list of files to be downloaded
    """
    layers = (
        # Embedding layer
        [0] +
        # Transformer layers
        list(range(2, 46)) +
        # Final normalization layer and readout layer
        [47, 48]
    )

    return (
        # Vocabulary and configs
        ['20B_tokenizer.json', 'configs/20B.yml', 'latest'] +
        # Layer checkpoints
        [f'global_step150000/layer_{i :02d}-model_{p :02d}-model_states.pt' for i in layers for p in range(2)] +
        # Empty states (not used)
        [f'global_step150000/mp_rank_{i :02d}_model_states.pt' for i in range(8)]
    )


def download():
    """
    ## Download all checkpoint files
    """

    # Get files to download
    files = get_files_to_download()

    # Iterate
    for i, f in monit.enum('Download All', files):
        # Log
        logger.log(['Downloading ', (f'{i + 1 :3d}/{len(files)}', Text.meta), ': ', (f, Text.value)])
        # Download
        download_file(CHECKPOINTS_URL + f, CHECKPOINTS_DOWNLOAD_PATH / f)


def get_checkpoint_files(n_layers: int = 44):
    """
    ## Generator to get checkpoint file names in order

    :param n_layers: is the number of transformer layers
    :return: the layers as a generator
    """

    # Embedding layer
    yield 'layer_00-model_00-model_states.pt', 'layer_00-model_01-model_states.pt'

    # Skip permutation layer (out implementation doesn't need this)
    yield None

    # Transformer layers
    for i in range(n_layers):
        yield f'layer_{i + 2 :02d}-model_00-model_states.pt', f'layer_{i + 2 :02d}-model_01-model_states.pt'

    # Skip permutation layer (out implementation doesn't need this)
    yield None

    # Final normalization layer
    yield 'layer_47-model_00-model_states.pt', 'layer_47-model_01-model_states.pt'

    # Readout layer
    yield 'layer_48-model_00-model_states.pt', 'layer_48-model_01-model_states.pt'


def load_checkpoint_files(files: Tuple[str, str]):
    """
    ### Load a pair of checkpoint files

    :param files: pair of files to load
    :return: the loaded parameter tensors
    """
    checkpoint_path = CHECKPOINTS_DOWNLOAD_PATH / 'global_step150000'
    with monit.section('Load checkpoint'):
        data = [torch.load(checkpoint_path / f) for f in files]

    return data


def merge_params_dim_0(param: Union[nn.Parameter, torch.Tensor], key: str, p1: Dict[str, torch.Tensor],
                       p2: Dict[str, torch.Tensor]):
    """
    ### Load a parameter by merging the partitions along first dimension

    :param param: is the parameter
    :param key: is the name of the parameter
    :param p1: first partition dictionary
    :param p2: second partition dictionary
    """
    w1, w2 = p1[key], p2[key]
    param.data[:w1.shape[0]] = w1
    param.data[w1.shape[0]:] = w2


def merge_params_dim_1(param: Union[nn.Parameter, torch.Tensor], key: str, p1: Dict[str, torch.Tensor],
                       p2: Dict[str, torch.Tensor]):
    """
    ### Load a parameter by merging the partitions along second dimension

    :param param: is the parameter
    :param key: is the name of the parameter
    :param p1: first partition dictionary
    :param p2: second partition dictionary
    """
    w1, w2 = p1[key], p2[key]
    param.data[:, :w1.shape[1]] = w1
    param.data[:, w1.shape[1]:] = w2


def merge_params_duplicate(param: Union[nn.Parameter, torch.Tensor], key: str, p1: Dict[str, torch.Tensor],
                           p2: Dict[str, torch.Tensor]):
    """
    ### Load an un-partitioned parameter

    This does a sanity check to make use both partitions are the same

    :param param: is the parameter
    :param key: is the name of the parameter
    :param p1: first partition dictionary
    :param p2: second partition dictionary
    """
    w1, w2 = p1[key], p2[key]

    diff = sum((w1 - w2) ** 2).item()
    assert diff < 1e-4, f'The partitions do not match: {key}'

    param.data[:] = (w1 + w2) / 2.


def merge_params_sum(param: Union[nn.Parameter, torch.Tensor], key: str, p1: Dict[str, torch.Tensor],
                     p2: Dict[str, torch.Tensor]):
    """
    ### Load biases that are partitioned which gets added on reduce

    :param param: is the parameter
    :param key: is the name of the parameter
    :param p1: first partition dictionary
    :param p2: second partition dictionary
    """
    w1, w2 = p1[key], p2[key]

    param.data[:] = w1 + w2


#
if __name__ == '__main__':
    download()
