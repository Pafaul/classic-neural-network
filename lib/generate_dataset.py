import itertools
import random

def sum_of_squares(inputs):
    y = sum([x**2 for x in inputs])
    return y

def linear_combination(inputs):
    y = sum(inputs)
    return y


def generate_dataset(func, *inputs, randomize=True):
    input_data = itertools.product(*inputs)
    used_data = None
    if randomize == True:
        used_data = sorted(input_data, key=lambda k: random.random())
    else:
        used_data = input_data
    for input_x in used_data:
    # for input_x in input_data:
        value = func(input_x)
        if value <= 1:
            yield [input_x, value]