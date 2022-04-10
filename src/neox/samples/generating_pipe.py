import fairscale
import torch
from torch import nn

from labml import monit
from neox.utils import print_token_outputs, load_layers, get_tokens, print_tokens
from neox.utils.cache import get_cache

LAYERS = None
PROMPT = 'Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German'


def infer(model, ids, device):
    with torch.no_grad():
        x = torch.tensor(ids)[None, :].to(device)
        x = model(x)

    print_token_outputs(ids, x)

    return x[0].max(dim=-1)[1].tolist()


def generate():
    cache = get_cache()
    cache.set('use_cache', True)

    layers = load_layers(LAYERS)

    with monit.section('Sequential'):
        model = nn.Sequential(*layers)

    with monit.section('Pipe'):
        n_layers = len(layers)
        n_gpus = 4
        balance = []
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
        for i in range(n_gpus):
            balance.append((n_layers - sum(balance)) // (n_gpus - i))
        pipe_model = fairscale.nn.Pipe(model,
                                       balance=balance,
                                       devices=devices)

    ids = get_tokens(PROMPT)

    cache.set('state_ids', (None, 1))
    next_token = infer(pipe_model, ids, pipe_model.devices[0])[-1]

    full_tokens = ids + [next_token]

    for i in range(1, 100):
        cache.set('state_ids', (i, i + 1))
        next_token = infer(pipe_model, [next_token], pipe_model.devices[0])[-1]
        full_tokens += [next_token]
        print_tokens(full_tokens, [full_tokens])


if __name__ == '__main__':
    generate()
