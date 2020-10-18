import numpy as np
from activation_function import ActivationFunction

class CalculationUnit:
    def run(self, input_value):
        pass

    def initialize(self, input_value):
        pass

    def correction(self, correction_value, correction_k):
        pass

class Neuron(CalculationUnit):
    def __init__(self, input_size: int, activation_function: ActivationFunction):
        self.input_size = input_size
        self.activation_function = activation_function
        self.input = np.zeros([self.input_size, 1], 'float64')
        self.weights = np.zeros([self.input_size, 1], 'float64')
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

    def correction(self, correction_value, correction_k=0):
        correction_value = np.multiply(correction_value, self.output)
        self.weights += correction_value

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

    def correction(self, correction_value, correction_k):
        weights_delta = correction_value*correction_k
        for neuron, weights_delta in zip(self.neurons, weights_delta):
            neuron.correction(weights_delta)

    def calc_diffs(self, error):
        return np.multiply(error, np.multiply(self.output, (1-self.output)))

    def calc_deltas(self, error):
        deltas = np.zeros([self.input_size, 1], 'float64')
        for neuron_error, neuron in zip(error, self.neurons):
            deltas += np.multiply(neuron_error, neuron.weights[1:])
                
        return deltas
            

class NeuralNetwork(CalculationUnit):
    def __init__(self, neural_network_structure: list, activation_function: ActivationFunction, learning_rate: float):
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
        self.learning_rate = learning_rate

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

    def correction(self, correction_value, correction_k=0):
        layers_diff = []

        output_layer = self.layers[-1]
        previous_layer_error = (correction_value-self.output)
        previous_layer_diff = output_layer.calc_diffs(previous_layer_error)
        layers_diff.append(previous_layer_diff)

        layer_error = None
        layer_diff = None

        for layer in reversed(self.layers[:-1]):
            layer_error = layer.calc_diffs(previous_layer_diff)
            layer_diff = layer.calc_deltas(layer_error)

            previous_layer_error = layer_error
            previous_layer_diff = layer_diff

            layers_diff.append(previous_layer_diff)

        for layer, diff in zip(self.layers, reversed(layers_diff)):
            layer.correction(diff, self.learning_rate)
