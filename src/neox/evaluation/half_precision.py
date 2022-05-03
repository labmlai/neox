import torch
from labml import monit
from torch import nn

from neox.evaluation import run_eval_harness
from neox.utils import load_layers

if __name__ == '__main__':
    layers = load_layers(None)

    with monit.section('Sequential'):
        model = nn.Sequential(*layers).half().to(torch.device('cuda:0'))

    print(run_eval_harness(model, 'half_precision', [], torch.device('cuda:0')))
