import numpy as np
import matplotlib.pyplot as plt

from lib.activation_function import Sigmoid,LeakyReLU,SigmoidParam
from lib.generate_dataset import generate_dataset,sum_of_squares,linear_combination
from lib.nn import Neuron
from lib.nn import Layer
from lib.nn import NeuralNetwork

def real_data_example():
    s = Sigmoid()
    sp = SigmoidParam()
    leaky = LeakyReLU()
    network = NeuralNetwork([4,16,8,1], sp, .01)
    print(str(network))
    print('Generating training data set. Please wait, it could take a lot of time...')
    using_function = sum_of_squares
    data_set = generate_dataset(
        using_function,
        np.arange(0, 1, 0.025),
        np.arange(0, 1, 0.025),
        np.arange(0, 1, 0.025),
        np.arange(0, 1, 0.025)
    )
    print('Running network traing')
    counter = 0
    error = 0
    epoch = 0
    counter_history = []
    error_history = []
    for x, y in data_set:
        x = np.array(x)
        network.run(x)
        error += (y - network.output)**2
        counter += 1
        network.correction(np.array([y]))
        if counter % 1000 == 0 and counter != 0:
            print(str(network))
            print('required output: ', y)
            error_history.append((error)/counter)
            counter_history.append(epoch)
            epoch += 1
            counter = 0
            error = 0
    plt.plot(np.array(counter_history), np.array(error_history))
    plt.grid(True)
    plt.show()

    print('Running test examination')
    x1 = np.arange(0, 1, 0.01)
    data_set = [None, None]
    data_set[0] = [[x1_1, 0, 0, 0] for x1_1 in x1]
    data_set[1] = [using_function(x_1) for x_1 in data_set[0]]
    
    output_history = []
    counter_history = []
    error = 0
    counter = 0
    for x, y in zip(data_set[0], data_set[1]):
        x = np.array(x)
        network.run(x)
        output_history.append(network.output)
        counter_history.append(counter)
        counter += 1
        error += (y - network.output)**2

    print(error)
    plt.plot(np.array(counter_history), np.array(output_history))
    plt.show()

real_data_example()