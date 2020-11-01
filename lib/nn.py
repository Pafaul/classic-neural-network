import numpy as np
from activation_function import ActivationFunction

class CalculationUnit:
    def run(self, input_value):
        pass

    def initialize(self, input_value):
        pass

    def correction(self, learning_rate, correction_k):
        pass

class Neuron(CalculationUnit):
    def __init__(self, input_size: int, activation_function: ActivationFunction):
        self.input_size = input_size
        self.activation_function = activation_function
        self.input = np.zeros([self.input_size], 'float64')
        self.weights = np.zeros([self.input_size], 'float64')
        self.output = 0
        self.v = 0
        self.local_gradient = 0

    def initialize(self, input_value=None):
        if (input_value is not None):
            self.weights = np.copy(input_value)
        else:
            self.weights = (np.random.rand(self.input_size)-0.5)*2
        return self.weights

    def run(self, input_value: np.array):
        self.input = np.copy(input_value)
        self.v = np.dot(self.input.transpose(), self.weights)
        self.output = self.activation_function.calc(self.v)
        return self.output

    def correction(self, learning_rate, correction_k=0):
        for weight_index in range(len(self.weights)):
            self.weights[weight_index] += learning_rate*self.local_gradient*self.input[weight_index]

    def calculate_derivative(self):
        return self.activation_function.derivative(self.v)

    def calculate_local_gradient(self, value):
        self.local_gradient = self.calculate_derivative()*value
        return self.local_gradient

class Layer(CalculationUnit):
    def __init__(self, input_size: int, output_size: int, activation_function: ActivationFunction, output_layer = False):
        self.neurons = [Neuron(input_size+1, activation_function) for x in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size
        self.input = np.zeros([self.input_size+1])
        self.output = np.zeros([self.output_size])
        self.output_layer = output_layer

    def __str__(self):
        description_string = ''
        description_string += 'Layer input size: ' + str(self.input_size) + '\n'
        description_string += 'Layer output size: ' + str(self.output_size) + '\n'
        description_string += 'Layer neuron count: ' + str(len(self.neurons)) + '\n'
        description_string += 'Layer output: ' + '\n' + str(self.output)
        return description_string

    def mark_as_output_layer(self):
        self.output_layer = True

    def initialize(self, input_value=None):
        for neuron in self.neurons:
            neuron.initialize(input_value)
        return self.neurons

    def run(self, input_value: np.array):
        self.input = np.insert(input_value, 0, 1).reshape([input_value.size+1])
        
        for neuron_index in range(self.output_size):
            self.output[neuron_index] = self.neurons[neuron_index].run(self.input)
        return self.output

    def correction(self, learning_rate):
        for neuron in self.neurons:
            neuron.correction(learning_rate)

    def calculate_local_gradient(self, previous_layer, training_data=None):
        if self.output_layer:
            for neuron_index in range(len(self.neurons)):
                self.neurons[neuron_index].calculate_local_gradient(training_data[neuron_index] - self.neurons[neuron_index].output)
        else:
            for neuron_index in range(len(self.neurons)):
                neuron_local_gradient = 0
                for neuron_previous_layer in previous_layer.neurons:
                    neuron_local_gradient += neuron_previous_layer.local_gradient * neuron_previous_layer.weights[neuron_index]
                self.neurons[neuron_index].calculate_local_gradient(neuron_local_gradient)


class NeuralNetwork(CalculationUnit):
    def __init__(self, neural_network_structure: list, activation_function: ActivationFunction, learning_rate: float):
        self.neural_network_structure = neural_network_structure

        self.layers = []
        for input_size, output_size in zip(neural_network_structure[:-1], neural_network_structure[1:]):
            self.layers.append(Layer(
                input_size,
                output_size,
                activation_function))

        self.layers[-1].mark_as_output_layer()

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
        description_string += 'Network output: ' + str(self.output)
        return description_string

    def run(self, input_value: np.array):
        self.input = np.copy(input_value)

        for layer in self.layers:
            result = layer.run(input_value)
            input_value = result

        self.output = np.copy(self.layers[-1].output) 
        return np.copy(self.output)

    def correction(self, required_outputs, correction_k=0):
        output_layer = self.layers[-1]
        output_layer.calculate_local_gradient(None, required_outputs)

        previous_layer = output_layer
        for layer in reversed(self.layers[:-1]):
            layer.calculate_local_gradient(previous_layer)
            previous_layer = layer

        for layer in self.layers:
            layer.correction(self.learning_rate)

    def dump_net_to_file(self, filename='network_structure.net'):
        content = ''
        content += ' '.join([str(x) for x in self.neural_network_structure]) + '\n'
        for layer in self.layers:
            for neuron in layer.neurons:
                content += ' '.join([str(x) for x in neuron.weights]) + '\n'

        with open(filename, 'w') as f:
            f.write(content)

    def recreate_net(self, filename):
        def iterator(iterable: list):
            for item in iterable:
                yield item

        def get_weights_from_line(line):
            return [float(x) for x in line.strip().split(' ')]

        new_network = None
        lines = None
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        network_structure = list(int(x) for x in lines[0].split(' '))
        weights = iterator(lines[1:])

        new_network = NeuralNetwork(network_structure)
        for layer in new_network.layers:
            for neuron in layer.neurons:
                neuron.set_weights(np.array(get_weights_from_line(next(weights)), 'float64'))

        return new_network