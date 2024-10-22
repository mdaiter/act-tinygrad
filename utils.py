from typing import Callable

import numpy as np
from tinygrad import Tensor, dtypes

def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return Tensor.relu
    if activation == "gelu":
        return Tensor.gelu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
