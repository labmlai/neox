import torch

from labml import monit
from neox.utils import print_token_outputs, load_layers, get_tokens, print_tokens
from neox.utils.cache import get_cache

LAYERS = None
PROMPT = 'Einstein was born in the German Empire, but moved to Switzerland in 1895, forsaking his German'


def infer(layers, ids, device):
    offload = torch.device('cpu')
    s = torch.cuda.Stream()
    with torch.no_grad():
        x = torch.tensor(ids)[None, :].to(device)
        for layer in layers:
            layer.to(device)
            x = layer(x)
            with torch.cuda.stream(s):
                layer.to(offload)

    print_token_outputs(ids, x)

    return x[0].max(dim=-1)[1].tolist()


def generate():
    cache = get_cache()
    cache.set('use_cache', True)

    layers = load_layers(LAYERS)

    device = torch.device('cuda:0')
    ids = get_tokens(PROMPT)

    cache.set('state_ids', (None, 1))
    with monit.section('Infer'):
        next_token = infer(layers, ids, device)[-1]

    full_tokens = ids + [next_token]

    for i in range(1, 100):
        cache.set('state_ids', (i, i + 1))
        with monit.section('Infer'):
            next_token = infer(layers, [next_token], device)[-1]
        full_tokens += [next_token]
        print_tokens(full_tokens, [full_tokens])


if __name__ == '__main__':
    generate()
