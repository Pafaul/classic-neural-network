import numpy as np
import matplotlib.pyplot as plt

from lib.activation_function import Sigmoid,LeakyReLU,SigmoidParam,Th
from lib.generate_dataset import generate_dataset,sum_of_squares
from lib.nn import NeuralNetwork
from lib.teacher import Teacher


def main():
    activation_function = SigmoidParam(6)
    network = NeuralNetwork([4,20,20,1], activation_function, .0005)
    function_for_approximation = sum_of_squares
    teacher = Teacher()
    print(str(network))
    print('Generating training dataset...')
    dataset = generate_dataset(
        function_for_approximation,
        np.arange(0, 1, 0.025),
        np.arange(0, 1, 0.025),
        np.arange(0, 1, 0.025),
        np.arange(0, 1, 0.025)
    )

    print('Strating network training.\nSome weird symbols will run on screen during training.\nYou can go and get yourself a cup of tea.')
    counters, errors = teacher.train(network, dataset, acceptable_error=0.03)

    plt.plot(np.array(counters), np.array(errors))
    plt.grid(True)
    plt.show()

    print('Running test examination')
    test_dataset = generate_dataset(
        function_for_approximation,
        np.arange(0,1, 0.05),
        np.arange(0,1,1),
        np.arange(0,1,1),
        np.arange(0,1,1),
        randomize=False
    )
    sum_error, counters, output = teacher.test(network, test_dataset)
    print('Total error: ', sum_error)

    ideal_x = np.arange(0,1,0.05)
    ideal_y = np.array([x**2 for x in ideal_x])
    plt.plot(np.array(counters), ideal_y)
    plt.plot(np.array(counters), np.array(output), 'ro')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()