from nn import NeuralNetwork
import numpy as np

class Teacher:

    def train(self, neuralNetwork: NeuralNetwork, dataset, acceptable_error=0, examples_in_epoch=1000):
        counter = 0
        error = 0
        epoch = 0
        counter_history = []
        error_history = []

        for x, y in dataset:
            x = np.array(x)
            neuralNetwork.run(x)
            error += (y - neuralNetwork.output)**2
            counter += 1
            neuralNetwork.correction(np.array([y]))
            if counter % examples_in_epoch == 0 and counter != 0:
                relative_error = error/counter
                error_history.append(relative_error)
                counter_history.append(epoch)
                epoch += 1
                print("Epoch: ", epoch, " | relative error: ", relative_error)
                if (error <= acceptable_error):
                    print('Learinig finished at epoch: ', epoch)
                    break

                counter = 0
                error = 0
        return counter_history,error_history

    def test(self, neuralNetwork: NeuralNetwork, dataset):
        error = 0
        counter = 0
        counter_history = []
        output_history = []
        error = 0
        counter = 0

        for x, y in dataset:
            x = np.array(x)
            neuralNetwork.run(x)
            output_history.append(neuralNetwork.output)
            counter_history.append(counter)
            counter += 1
            error += (y - neuralNetwork.output)**2

        return [error, counter_history, output_history]
