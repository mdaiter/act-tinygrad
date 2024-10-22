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

def clip_grad_norm_(parameters, max_norm) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    
    max_norm = float(max_norm)
    
    if len(parameters) == 0:
        return Tensor(0.0)
    
    total_norm = Tensor.zeros((), dtype=dtypes.float32)
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.flatten().square().sum().realize()
            total_norm += param_norm
            total_norm = total_norm.contiguous().realize()
    
    total_norm = (total_norm + 1e-12).sqrt()
    print(f'total_norm: {total_norm.numpy()}')
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = Tensor.minimum(clip_coef, Tensor.ones_like(clip_coef)).contiguous().realize().item()
    print(f'clip_coef: {clip_coef}')
    
    for p in parameters:
        if p.grad is not None:
            p.grad *= clip_coef
    
    return total_norm
