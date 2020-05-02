import operator
from functools import reduce
from typing import Tuple


def conv_output_size(in_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return (in_size - kernel_size + 2 * padding) // stride + 1


def conv_output_shape(input_shape: Tuple[int, int, int],
                      out_channels: int,
                      kernel_size: int,
                      stride: int = 1,
                      padding: int = 0) -> Tuple[int, int, int]:
    return (
        out_channels,
        conv_output_size(input_shape[1], kernel_size=kernel_size, stride=stride, padding=padding),
        conv_output_size(input_shape[2], kernel_size=kernel_size, stride=stride, padding=padding)
    )


def prod(iterable):
    return reduce(operator.mul, iterable, 1)
