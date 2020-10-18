import numpy as np
from lib.activation_function import Sigmoid
from lib.generate_dataset import generate_dataset,sum_of_squares
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
    network = NeuralNetwork([2,2,1], s, 0.1)
    print(str(network))
    for _ in range(10000):
        network.run(np.array([[-10.], [-10.]]))
        print(network.output)
        network.correction(np.array([1]))


full_test()
