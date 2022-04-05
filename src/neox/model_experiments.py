"""
10_000 values cause inf in float16. This happens in attention matrix computation.
But clipping causes large errors and doesn't work.
"""

import torch


def _value(attn: torch.Tensor, v: torch.Tensor):
    """
    output = torch.einsum('bijh,bjhk->bihk', attn, v)
    """
    print('value', attn.dtype, v.dtype)
    batch_size, i, j, n_head = attn.shape

    attn = attn.permute(0, 3, 1, 2).view(batch_size * n_head, i, j)  # bh, i, j
    v = v.permute(0, 2, 1, 3).view(batch_size * n_head, j, -1)  # bh, j, d_k

    # output = torch.bmm(attn, v)  # bh, i, k

    output = torch.empty((batch_size * n_head, i, v.shape[-1]), dtype=v.dtype, device=v.device)

    output = torch.baddbmm(output, attn, v,
                           beta=0.0,
                           alpha=1.0)

    output = output.view(batch_size, n_head, i, -1)

    return output.permute(0, 2, 1, 3)


def _attn_baddbmm(q: torch.Tensor, k: torch.Tensor):
    """
    attn = self._attn_baddbmm(q, k)
    instead of attn = torch.einsum('bihk,bjhk->bijh', q, k)
    """
    print('attn', q.dtype, k.dtype)
    batch_size, seq_len, n_head, d_k = q.shape

    q = q.permute(0, 2, 1, 3).view(-1, seq_len, d_k)
    k = k.permute(0, 2, 1, 3).view(-1, seq_len, d_k)

    attn = torch.empty((batch_size * n_head, seq_len, seq_len), dtype=q.dtype, device=q.device)

    attn = torch.baddbmm(attn, q, k.transpose(-1, -2),
                         beta=0.0,
                         alpha=1.0)

    attn = attn.view(batch_size, n_head, seq_len, seq_len)
    attn = attn.permute(0, 2, 3, 1)

    return attn
