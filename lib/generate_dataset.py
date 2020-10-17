import math
import numpy as np
import itertools

def sum_of_squares(inputs):
    y = sum([x**2 for x in inputs])
    return y

def generate_dataset(func, *inputs):
    input_data = itertools.product(*inputs)
    for input_x in input_data:
        yield [input_x, func(input_x)]