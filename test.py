import numpy as np
import matplotlib.pyplot as plt

from lib.activation_function import Sigmoid,LeakyReLU,SoftPlus,Equal
from lib.generate_dataset import generate_dataset,sum_of_squares,linear_combination
from lib.nn import Neuron
from lib.nn import Layer
from lib.nn import NeuralNetwork

def sigmoid_test():
    print('Sigmoid test')
    s = Sigmoid()
    for x in np.arange(-1, 1, 0.1):
        print('s(', x, '): ', s.calc(x))

def data_generation_test():
    print('Dataset test')
    for data in generate_dataset(sum_of_squares, np.arange(-1, 1, 0.5), np.arange(0, 2, 0.5)):
        print(data)

def neuron_test():
    print('Neuron test')
    s = Sigmoid()
    neuron = Neuron(2, s)
    neuron.initialize(np.array([2,1]))
    neuron.run(np.array([[2],[2]]))
    print('Neuron inputs: ', neuron.input)
    print('Neuron weights: ', neuron.weights)
    print('Neuron output: ', neuron.output)

def layer_test():
    print('Layer test')
    s = Sigmoid()
    l = Layer(2, 2, s)
    l.initialize(np.array([[0], [1], [1]]))
    l.run(np.array([[2], [2]]))
    print(str(l))

def network_test():
    print('Network test')
    s = Sigmoid()
    network = NeuralNetwork([2, 2, 3, 1], s, 0.1)
    network.run(np.array([[2], [2]]))
    print(str(network))
    for layer in network.layers:
        print(str(layer))

def full_test():
    print('Network run test')
    s = Sigmoid()
    leaky = LeakyReLU()
    eq = Equal()
    network = NeuralNetwork([1,2,1], s, 1)
    print(str(network))
    for _ in range(10000):
        network.run(np.array([[10.0]]))
        print(network.output)
        network.correction(np.array([.9]))


def real_data_example():
    s = Sigmoid()
    sp = SigmoidParam(8)
    leaky = LeakyReLU()
    th=Th()
    network = NeuralNetwork([4,40,30,20,1], th, .0005)
    network.dump_net_to_file()
    print(str(network))
    print('Generating training data set. Please wait, it could take a lot of time...')
    using_function = sum_of_squares
    data_set = generate_dataset(
        using_function,
        np.arange(-1, 1, 0.025),
        np.arange(-1, 1, 0.025),
        np.arange(-1, 1, 0.025),
        np.arange(-1, 1, 0.025)
    )
    print('Running network traing')
    counter = 0
    error = 0
    epoch = 0
    counter_history = []
    error_history = []

    required_error = 0.005
    for x, y in data_set:
        x = np.array(x)
        network.run(x)
        error += (y - network.output)**2
        counter += 1
        network.correction(np.array([y]))
        if counter % 1000 == 0 and counter != 0:
            relative_error = error/counter
            error_history.append(relative_error)
            counter_history.append(epoch)
            epoch += 1
            print("Epoch: ", epoch, " | relative error: ", relative_error)
            if (error/counter <= required_error):
                print('Learinig finished at epoch: ', epoch)
                break

            counter = 0
            error = 0
            
    plt.plot(np.array(counter_history), np.array(error_history))
    plt.grid(True)
    plt.show()

    print('Running test examination')
    x1 = np.arange(-1, 1, 0.01)
    data_set = [None, None]
    data_set[0] = [[x1_1, 0, 0, 0] for x1_1 in x1]
    data_set[1] = [using_function(x_1) for x_1 in data_set[0]]
    data_set = generate_dataset(
        using_function,
        np.arange(-1, 1, 0.025),
        np.arange(-1, 1, 0.025),
        np.arange(-1, 1, 0.025),
        np.arange(-1, 1, 0.025)
    )
    reference_values = [x1_1**2 for x1_1 in x1]
    
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
    plt.plot(np.array(counter_history), np.array(reference_values))
    plt.plot(np.array(counter_history), np.array(output_history), 'ro')
    plt.show()

    try:
        network.dump_net_to_file()
    except:
        print('cannot dump')
