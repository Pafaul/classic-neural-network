import itertools
import random

def sum_of_squares(inputs):
    y = sum([x**2 for x in inputs])
    return y

def linear_combination(inputs):
    y = sum(inputs)
    return y


def generate_dataset(func, *inputs):
    input_data = itertools.product(*inputs)
    for input_x in sorted(input_data, key=lambda k: random.random()):
    # for input_x in input_data:
        value = func(input_x)
        if value <= 1:
            yield [input_x, value]