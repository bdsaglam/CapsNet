import operator
from functools import reduce
from typing import Tuple

import torch


def conv_output_size(in_size: int,
                     kernel_size: int,
                     stride: int = 1,
                     padding: int = 0) -> int:
    return (in_size - kernel_size + 2 * padding) // stride + 1


def conv_output_shape(input_shape: Tuple[int, int, int],
                      out_channels: int,
                      kernel_size: int,
                      stride: int = 1,
                      padding: int = 0) -> Tuple[int, int, int]:
    return (
        out_channels,
        conv_output_size(input_shape[1],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding),
        conv_output_size(input_shape[2],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)
    )


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = input_tensor / safe_norm
    output_tensor = squash_factor * unit_vector
    return output_tensor


def safe_norm(input_tensor: torch.Tensor,
              dim: int = -1,
              epsilon: float = 1e-7,
              keepdim: bool = False) -> torch.Tensor:
    squared_norm = (input_tensor ** 2).sum(dim, keepdim=keepdim)
    return torch.sqrt(squared_norm + epsilon)
