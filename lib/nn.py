import numpy as np
from activation_function import ActivationFunction

class CalculationUnit:
    def run(self, input_value):
        pass

    def initialize(self, input_value):
        pass

    def correction(self, correction_value, **args):
        pass

class Neuron(CalculationUnit):
    def __init__(self, input_size: int, activation_function: ActivationFunction):
        self.input_size = input_size
        self.activation_function = activation_function
        self.input = np.zeros([self.input_size, 1])
        self.weights = np.zeros([self.input_size, 1])
        self.output = 0

    def initialize(self, input_value=None):
        if (input_value is not None):
            self.weights = np.copy(input_value)
        else:
            self.weights = (np.random.rand(self.input_size, 1)-0.5)*2
        return self.weights

    def run(self, input_value: np.array):
        self.input = np.copy(input_value)
        self.output = self.activation_function.calc(np.dot(self.input.transpose(), self.weights))
        return self.output

    def correction(self, correction_value, **args):
        pass

class Layer(CalculationUnit):
    def __init__(self, input_size: int, output_size: int, activation_function: ActivationFunction):
        self.neurons = [Neuron(input_size+1, activation_function) for x in range(output_size)]     
        self.input_size = input_size
        self.output_size = output_size
        self.input = np.zeros([self.input_size+1, 1])
        self.output = np.zeros([self.output_size, 1])

    def __str__(self):
        description_string = ''
        description_string += 'Layer input size: ' + str(self.input_size) + '\n'
        description_string += 'Layer output size: ' + str(self.output_size) + '\n'
        description_string += 'Layer neuron count: ' + str(len(self.neurons)) + '\n'
        description_string += 'Layer output: ' + '\n' + str(self.output) + '\n'
        return description_string

    def initialize(self, input_value=None):
        for neuron in self.neurons:
            neuron.initialize(input_value)
        return self.neurons

    def run(self, input_value: np.array):
        self.input = np.insert(input_value, 0, 1).reshape([input_value.size+1, 1])
        
        for neuron_index in range(self.output_size):
            self.output[neuron_index] = self.neurons[neuron_index].run(self.input)
        return self.output

    def correction(self, correction_value, **args):
        pass

class NeuralNetwork(CalculationUnit):
    def __init__(self, neural_network_structure: list, activation_function: ActivationFunction):
        self.neural_network_structure = neural_network_structure

        self.layers = []
        for input_size, output_size in zip(neural_network_structure[:-1], neural_network_structure[1:]):
            self.layers.append(Layer(
                input_size,
                output_size,
                activation_function))

        for layer in self.layers:
            layer.initialize()

        self.input = np.zeros([self.neural_network_structure[0], 1])
        self.output = np.zeros([self.neural_network_structure[-1], 1])

        self.activation_function = activation_function

    def __str__(self):
        description_string = ''
        description_string += 'Network structure: ' + str(self.neural_network_structure) + '\n'
        description_string += 'Network input: ' + str(self.input) + '\n'
        description_string += 'Network output: ' + str(self.output) + '\n'
        return description_string

    def run(self, input_value: np.array):
        self.input = np.copy(input_value)

        for layer in self.layers:
            result = layer.run(input_value)
            input_value = result

        self.output = np.copy(self.layers[-1].output) 
        return np.copy(self.output)

    def correction(self, correction_value, **args):
        pass