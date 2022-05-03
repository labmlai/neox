"""
---
title: Test evaluations with a dummy model
summary: >
    Test the evaluation harness adapter with a dummy model
---

# Test evaluations with a dummy model
"""

import torch
from torch import nn

from neox.evaluation import run_eval_harness


class DummyModel(nn.Module):
    """
    ## Dummy model
    """

    def __init__(self, n_vocab: int):
        super().__init__()
        self.n_vocab = n_vocab

    def forward(self, x: torch.Tensor):
        return torch.randn(x.shape + (self.n_vocab,), dtype=torch.float, device=x.device)


if __name__ == '__main__':
    print(run_eval_harness(DummyModel(50_432), 'dummy', ['lambada'], torch.device('cpu')))
