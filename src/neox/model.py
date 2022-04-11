"""
---
title: GPT-NeoX Model Definition
summary: >
    This is the model definition of GPT-NeoX.
---

# GPT-NeoX Model

Here is the code for layers of GPT-NeoX model and the code to load
20B checkpoint.

The method `load_state` in the layers load the checkpoints of that layer.
The checkpoint loading helpers are on [`checkpoint.py`](checkpoint.html)
"""
import copy
import math
from typing import Dict, Optional, Set

import torch
from torch import nn
from torch.cuda.amp import autocast

from labml import monit
from labml.logger import inspect
from neox import checkpoint
from neox.utils.cache import get_cache


class Embedding(nn.Module):
    """
    ## Embedding layer

    This is a standard embeddings layer with code to load the checkpoint.
    """

    def __init__(self, n_vocab: int = 50_432, n_hidden: int = 6_144):
        """
        :param n_vocab: is the size of the vocabulary
        :param n_hidden: is the size of the embeddings
        """
        super().__init__()

        self.emb = nn.Embedding(n_vocab, n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the token ids of shape `[batch_size, seq_len]`
        """
        return self.emb(x)

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load embedding layer'):
            checkpoint.merge_params_dim_0(self.emb.weight, 'word_embeddings.weight', p1, p2)


class RoPE(torch.nn.Module):
    """
    ## Rotary Positional Embeddings

    GPT-NeoX uses [rotary positional embeddings (RoPE)](https://papers.labml.ai/paper/2104.09864).

    WE have annotated implementation of RoPE [here](https://nn.labml.ai/transformers/rope/index.html)
    with more notes the theory.
    """

    def __init__(self, d_rope: int, base: float = 10_000.):
        """
        :param d_rope: is the number of features for RoPE embeddings
        :param base: is the base for $\theta_i = 10000^{\frac{2(i-1)}{d}}$, which defaults to $10000$
        """
        super().__init__()

        # To store $\theta_i$ for the features
        self.theta = None
        # Cache $\cos m\theta_i$ and $\sin m\theta_i$
        self.cos_cached = None
        self.sin_cached = None

        # Base for $\theta_i = 10000^{\frac{2(i-1)}{d}}$
        self.base = base
        # Number of features for RoPE
        self.d_rope = d_rope

    @staticmethod
    def rotate_half(x: torch.Tensor):
        """
        ### Rotate the features

        $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., -x^{(\frac{d}{2})}]$
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        :param x: has shape `[batch, seq, n_heads, d_k]`
        :param offset: is the starting position of `x`. This is $\gt 0$ when we have
        cached the keys and queries of previous positions
        """

        # Get the actual sequence length
        seq_len = x.shape[1] + offset

        # Initialize $\theta$
        if self.theta is None:
            #  $\theta_i = 10000^{\frac{2(i-1)}{d}}$
            theta = 1.0 / (self.base ** (torch.arange(0, self.d_rope, 2).float() / self.d_rope))
            self.theta = theta.to(x.device).to(x.dtype)

        # Initialize $\cos m\theta_i$ and $\sin m\theta_i$ cache
        if (
                self.cos_cached is None or
                seq_len > self.cos_cached.shape[1] or
                self.cos_cached.device != x.device or
                self.cos_cached.dtype != x.dtype
        ):
            # Get position indexes $m$
            seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)
            # $m \theta_i$
            idx_theta = torch.einsum("s,d->sd", seq_idx, self.theta)
            # Concatenate so that for row $m$ we have
            #
            # $$[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$$
            idx_theta2 = torch.cat((idx_theta, idx_theta), dim=-1).to(x.device)

            # Calculate $\cos m\theta_i$ and $\sin m\theta_i$ in fp32
            with autocast(enabled=False):
                idx_theta2 = idx_theta2.float()
                self.cos_cached = idx_theta2.cos()[None, :, None, :]
                self.sin_cached = idx_theta2.sin()[None, :, None, :]

            # Cache them
            self.cos_cached = self.cos_cached.to(x.dtype)
            self.sin_cached = self.sin_cached.to(x.dtype)

        # Split the features. We apply RoPE to only `d_rope` features
        x_rope, x_pass = x[..., :self.d_rope], x[..., self.d_rope:]

        # Get the sin and cos values from the cache
        cos, sin = self.cos_cached[:, offset: seq_len], self.sin_cached[:, offset: seq_len]

        # RoPE embeddings
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        x_rope = (x_rope * cos) + (self.rotate_half(x_rope) * sin)

        # Concatenate with features that didn't get RoPE embeddings
        return torch.cat((x_rope, x_pass), dim=-1)


class AttentionLayer(nn.Module):
    """
    ## Attention layer
    """

    def __init__(self, n_hidden: int = 6_144, n_heads: int = 64, rope_percentage: float = 0.25,
                 mask_fill: float = -10_000.0):
        """
        :param n_hidden: the number of features in embeddings
        :param n_heads: the number of attention heads
        :param rope_percentage: percentage of features to add RoPE embeddings
        :param mask_fill: masking fill value for attention matrix
        """
        super().__init__()

        self.n_heads = n_heads
        self.mask_fill = mask_fill

        # Linear layer for query, key and value
        self.qkv_lin = nn.Linear(n_hidden, n_hidden * 3)
        # Final linear layer
        self.output = nn.Linear(n_hidden, n_hidden)

        # Number of features per head
        d_k = n_hidden // n_heads
        # RoPE embedding module
        self.rope = RoPE(int(d_k * rope_percentage))

        # Attention scaling factor
        self.scale = 1 / math.sqrt(d_k)

        # To cache causal mask
        self.causal_mask = None

        # Attention softmax module
        self.softmax = nn.Softmax(dim=-2)

    def _get_mask(self, attn: torch.Tensor):
        """
        #### Calculate the causal mask

        * `attn` has shape [batch_size, query_seq_len, key_seq_len, n_heads]
        """

        # Query and key lengths
        nq, nk = attn.shape[1:3]

        # Create mask
        if (
                self.causal_mask is None or
                self.causal_mask.shape[0] != nq or
                self.causal_mask.shape[1] != nk or
                self.causal_mask.device != attn.device
        ):
            self.causal_mask = torch.triu(attn.new_ones([nq, nk], dtype=torch.bool), 1 + nk - nq)

        # Return from cache
        return self.causal_mask[None, :, :, None]

    def forward(self, x: torch.Tensor):
        """
        :param x: has shape `[batch_size, seq_len, n_hidden]`
        """
        # Get query, key and value embeddings (all concatenated).
        # The last dimension size will change from n_hidden -> `3 x n_hidden`
        qkv = self.qkv_lin(x)

        # Split into heads by changing the shape to `[batch_size, seq_len, n_heads, 3 * d_k]`
        qkv = qkv.view(*qkv.shape[:-1], self.n_heads, -1)
        # Split into query, key and value each of shape `[batch_size, seq_len, n_heads, 3 * d_k]`
        q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)

        # If we are caching the states of previous tokens
        if get_cache().get('use_cache', False):
            # Get the state id's. We use to retrieve previous states and store the next states
            prev_state_id, next_state_id = get_cache().get('state_ids')
            # If there's cache
            if prev_state_id is not None:
                # Get the past keys and values. These will have shape `[batch_size, prev_seq_len, n_heads, d_k]`
                k_past, v_past = get_cache().pop(f'attn_kv_{prev_state_id}')
                # Offset of the current embeddings
                offset = k_past.shape[1]

                # Add RoPE embeddings
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)

                # Concatenate the past
                k = torch.cat([k_past, k], dim=1)
                v = torch.cat([v_past, v], dim=1)
            else:
                # Add RoPE embeddings
                q = self.rope(q)
                k = self.rope(k)

            # Save the current state
            get_cache().push(f'attn_kv_{next_state_id}', (k, v))
        else:
            # No cache - simply add RoPE embeddings
            q = self.rope(q)
            k = self.rope(k)

        # Disable auto-casting to fp16 for attention computation
        with autocast(enabled=False):
            if q.dtype == torch.float16:
                # Convert to fp32 if the current dtype is fp16
                attn = torch.einsum('bihk,bjhk->bijh', q.float(), k.float())
            else:
                # Do not cast for bfloat
                attn = torch.einsum('bihk,bjhk->bijh', q, k)

            # Scale attention
            attn = attn * self.scale

            # Get causal mask
            mask = self._get_mask(attn)
            # Apply mask
            attn.masked_fill_(mask, self.mask_fill)

            # Attention softmax
            attn = self.softmax(attn)

        # Get attention weighted values
        output = torch.einsum('bijh,bjhk->bihk', attn.to(v.dtype), v)

        # Reshape from `[batch_size, seq_len, n_heads, d_k] to `[batch_size, seq_len, n_hidden]`
        output = output.reshape(*x.shape)

        # Final linear layer
        return self.output(output)


class FFNLayer(nn.Module):
    """
    ## Feedforward Network
    """

    def __init__(self, n_hidden: int = 6_144):
        """
        :param n_hidden: is the embedding size
        """
        super().__init__()

        # Expansion linear layer
        self.dense_h_h4 = nn.Linear(n_hidden, n_hidden * 4)
        # GELU activation
        self.activation = nn.GELU()
        # Contraction linear layer
        self.dense_h4_h = nn.Linear(n_hidden * 4, n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: has shape `[batch_size, seq_len, n_hidden]`
        """
        x = self.dense_h_h4(x)
        x = self.activation(x)
        x = self.dense_h4_h(x)

        return x


class TransformerLayer(nn.Module):
    """
    ## Transformer Layer
    """

    def __init__(self, n_hidden: int = 6_144, n_heads: int = 64):
        """
        :param n_hidden: is the embedding size
        :param n_heads: is the number of heads

        *Out implementation doesn't include dropout*.
        """
        super().__init__()

        # Layer normalization before attention
        self.pre_ln_attn = nn.LayerNorm(n_hidden)
        # Layer normalization before FFN
        self.pre_ln_ffn = nn.LayerNorm(n_hidden)

        # Attention layer
        self.attention = AttentionLayer(n_hidden, n_heads)
        # FFN layer
        self.ffn = FFNLayer(n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[batch_size, seq_len, n_hidden]`
        """

        # Residual connection
        residual = x
        # NeoX runs attention and feedforward network in parallel
        attn = self.attention(self.pre_ln_attn(x))
        ffn = self.ffn(self.pre_ln_ffn(x))
        # Add them and the residual connection
        return attn + ffn + residual

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load transformer layer'):
            # Attention output transform
            checkpoint.merge_params_duplicate(self.attention.output.bias, 'attention.dense.bias', p1, p2)
            checkpoint.merge_params_dim_1(self.attention.output.weight, 'attention.dense.weight', p1, p2)

            # Attention query, key and value transform
            checkpoint.merge_params_dim_0(self.attention.qkv_lin.bias, 'attention.query_key_value.bias', p1, p2)
            checkpoint.merge_params_dim_0(self.attention.qkv_lin.weight, 'attention.query_key_value.weight', p1, p2)

            # Layer norm before attention
            checkpoint.merge_params_duplicate(self.pre_ln_attn.bias, 'input_layernorm.bias', p1, p2)
            checkpoint.merge_params_duplicate(self.pre_ln_attn.weight, 'input_layernorm.weight', p1, p2)

            # FFN second transform
            checkpoint.merge_params_dim_0(self.ffn.dense_h_h4.bias, 'mlp.dense_h_to_4h.bias', p1, p2)
            checkpoint.merge_params_dim_0(self.ffn.dense_h_h4.weight, 'mlp.dense_h_to_4h.weight', p1, p2)

            # FFN first transform
            checkpoint.merge_params_duplicate(self.ffn.dense_h4_h.bias, 'mlp.dense_4h_to_h.bias', p1, p2)
            checkpoint.merge_params_dim_1(self.ffn.dense_h4_h.weight, 'mlp.dense_4h_to_h.weight', p1, p2)

            # Layer norm before FFN
            checkpoint.merge_params_duplicate(self.pre_ln_ffn.bias, 'post_attention_layernorm.bias', p1, p2)
            checkpoint.merge_params_duplicate(self.pre_ln_ffn.weight, 'post_attention_layernorm.weight', p1, p2)


class FinalNorm(nn.Module):
    """
    ## Final normalization layer
    """

    def __init__(self, n_hidden: int = 6_144):
        """
        :param n_hidden: is the embedding size
        """
        super().__init__()

        self.ln = nn.LayerNorm(n_hidden)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[batch_size, seq_len, n_hidden]`
        """
        return self.ln(x)

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load final normalization layer'):
            checkpoint.merge_params_duplicate(self.ln.bias, 'norm.bias', p1, p2)
            checkpoint.merge_params_duplicate(self.ln.weight, 'norm.weight', p1, p2)


class ReadoutLayer(nn.Module):
    """
    Readout layer
    """

    def __init__(self, n_hidden: int = 6_144, n_vocab: int = 50_432):
        """
        :param n_hidden: is the embedding size
        :param n_vocab: is the size of the vocabulary
        """
        super().__init__()

        self.linear = nn.Linear(n_hidden, n_vocab)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the embeddings of shape `[batch_size, seq_len, n_hidden]`
        """
        return self.linear(x)

    def load_state(self, p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor]):
        """
        Code to load the checkpoint
        """
        with monit.section('Load final linear layer'):
            checkpoint.merge_params_dim_0(self.linear.weight, 'final_linear.weight', p1, p2)


def get_layers(n_vocab: int = 50_432, n_hidden: int = 6_144, n_layers: int = 44, n_heads: int = 64,
               filter_layers: Optional[Set] = None, *,
               is_clone_layers: bool = False):
    """
    ### Generator to create layers

    The layers are generated in the same order as checkpoints.

    It gives `None` when a layer is not available; we use the layer indices as NeoX and there are two
    transformation layers we don't need in our implementation.

    :param n_vocab: is the number of tokens in the vocabulary
    :param n_hidden: is the number of features in the embeddings
    :param n_layers: is the number of transformer layers
    :param n_heads: is the number of attention heads
    :param filter_layers: are the set of layers to be used. All layers will be used if None.
        This is used to test smaller versions of the model with fewer layers
    :param is_clone_layers: specifies whether to clone the transformer layers (a bit faster)
    :return: the layers as a generator
    """

    # Embedding layer
    with monit.section('Embedding layer'):
        layer = Embedding(n_vocab, n_hidden)
    yield layer

    #
    yield None

    tl = None

    # Transformer layers
    for i in range(n_layers):
        # Yield `None` if we are skipping layers
        if filter_layers is not None and i not in filter_layers:
            yield None
            continue
        # Transformer layer
        with monit.section(f'Transformer Layer {i}'):
            if is_clone_layers:
                if tl is None:
                    tl = TransformerLayer(n_hidden, n_heads)
                    layer = tl
                else:
                    layer = copy.deepcopy(tl)
            else:
                layer = TransformerLayer(n_hidden, n_heads)
        yield layer

    #
    yield None

    # Final normalization layer
    with monit.section('Final norm layer'):
        layer = FinalNorm(n_hidden)
    yield layer
    # Readout layer
    with monit.section('Readout layer'):
        layer = ReadoutLayer(n_hidden, n_vocab)
    yield layer


def _test_load():
    """
    #### Test loading the model
    """

    # Iterate through the layers
    for layer, files in zip(get_layers(), checkpoint.get_checkpoint_files()):
        # Skip empty layers and empty checkpoints
        if layer is None or files is None:
            continue
        # Load checkpoint
        layer.load_state(*checkpoint.load_checkpoint_files(files))


def _test_eval():
    """
    ### Test the model

    Run the model on a sample text and observe the output.
    This loads the layers to CPU so it will run on any computer.

    This gets an accuracy of 336/573
    """
    from neox.utils import get_sample_tokens, print_token_outputs

    # [Get the encodings of a sample](utils.html)
    ids = get_sample_tokens(True)

    # No need to keep the computation graph for gradients
    with torch.no_grad():
        # Get the token ids and add a batch dimension
        x = torch.tensor(ids)[None, :]
        # Print the token ids
        inspect(x)

        # Iterate through the layers
        with monit.section("Layers"):
            for i, (layer, files) in enumerate(zip(get_layers(is_clone_layers=True), checkpoint.get_checkpoint_files())):
                # Skip empty layers and empty checkpoints
                if layer is None or files is None:
                    continue
                # Load layer checkpoint
                layer.load_state(*checkpoint.load_checkpoint_files(files))

                # Evaluate the layer
                with monit.section('Evaluate'):
                    x = layer(x)

                # Set progress
                monit.progress(i / 49)

    # Print the output of the model alongside the inputs
    print_token_outputs(ids, x)


#
if __name__ == '__main__':
    _test_eval()
