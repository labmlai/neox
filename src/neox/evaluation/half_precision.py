import torch
from labml import monit
from labml.logger import inspect
from torch import nn

from neox.evaluation import run_eval_harness
from neox.utils import load_layers

if __name__ == '__main__':
    layers = load_layers(None)

    with monit.section('Sequential'):
        model = nn.Sequential(*layers).half().to(torch.device('cuda:0'))

    inspect(run_eval_harness(model, 'half_precision', []), _expand=True, _n=-1)
