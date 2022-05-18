import torch
from torch import nn

from labml import monit
from neox.evaluation import run_eval_harness
from neox.utils import LayerGenerator

if __name__ == '__main__':
    device = torch.device('cuda:0')
    layers = list(LayerGenerator(is_clone_layers=True,
                                 filter_layers=None,
                                 dtype=torch.float16,
                                 device=device
                                 ).load())

    with monit.section('Sequential'):
        model = nn.Sequential(*layers).half().to()

    print(run_eval_harness(model, 'half_precision', [], device))
