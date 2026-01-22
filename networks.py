import math
from collections import deque
from itertools import chain
from typing import Callable, Optional, Union, Literal

import numpy as np
import tinygrad
from tinygrad import Tensor, nn, dtypes
from tinygrad import Variable


from utils import *

class ACTSinusoidalPositionEmbedding2d:
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = Tensor.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1).cast(dtype=dtypes.float32)
        x_range = not_mask.cumsum(2).cast(dtype=dtypes.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as 
        # all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = Tensor(self._temperature ** (
            2 * (np.arange(self.dimension, dtype='f') // 2) / self.dimension
        ))

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        print(x_range)
        print(y_range)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        x_range_sin = x_range[..., 0::2].sin()
        x_range_cos = x_range[..., 1::2].cos()
        y_range_sin = y_range[..., 0::2].sin()
        y_range_cos = y_range[..., 1::2].cos()
        print(f'x_range[..., 0::2].sin(): {x_range_sin}')
        print(f'x_range[..., 1::2].cos(): {x_range_cos}')
        pos_embed_x = x_range_sin.stack(x_range_cos, dim=-1).flatten(3)
        pos_embed_y = y_range_sin.stack(y_range_cos, dim=-1).flatten(3)
        pos_embed = pos_embed_y.cat(pos_embed_x, dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)], 
        dtype='f'
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return Tensor(sinusoid_table).float()

class MultiheadAttention:
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "n_state must be divisible by n_head"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scaling = self.head_dim ** -0.5        
        self.dropout = dropout

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, training: bool = True):
        batch_size, tgt_len, _ = q.shape
        src_len = k.shape[1]

        # Apply linear transformations
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # Reshape and transpose for multi-head attention
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scaling

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            print(f'(q,k,v): {q.shape}, {k.shape}, {v.shape}')
            print(f'(key_padding_mask): {key_padding_mask.shape}')
            # Reshape and expand key_padding_mask to match attn_scores dimensions
            key_padding_mask = key_padding_mask.squeeze(1).squeeze(1)  # Remove extra dimensions
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # Add dimensions for heads and query length
            key_padding_mask = key_padding_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))   
    
        # Apply softmax to get attention weights
        attn_weights = attn_scores.softmax(axis=-1)

        # Apply dropout
        if self.dropout > 0:
            attn_weights = attn_weights.dropout(p=self.dropout)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape and combine heads
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, tgt_len, self.embed_dim)

        # Final projection
        attn_output = self.out(attn_output)

        return attn_output


