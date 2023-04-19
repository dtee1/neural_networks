import numpy as np
import scipy.special
import numpy
import matplotlib.pyplot
from tkinter import Tk
from tkinter.filedialog import askdirectory
import argparse


class neuralNet:
    """
    A 3 layer neural network for single digit classification

    :param input_nodes(int): Number of input nodes, 784 nodes which corresponds to a 28x28 image
    :param hidden_nodes(int): Number of nodes in each hidden layer.
    :param learning_rate(float): Learning rates used to update weights during training

    This neural network(nn) randomly initializes the weights using a relation including the specified nodes for the hidden layer
    Also, this nn uses the sigmoid activation function and backpropagation to update the weights during training

    The 'train' method trains the network while the query method is used to predict given images.
    """

    def __init__(self, input_nodes, hidden_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = 10
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        self.wih = np.random.normal(
            0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes)
        )
        self.who = np.random.normal(
            0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes)
        )
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs, targets):
        """
        Train the neural network on single digits from the MNIST dataset

        Args:
            inputs (numpy.ndarray): Input features
            targets (numpy.ndarray): Target label
        """
        input = np.array(inputs, ndmin=2).T
        target = np.array(targets, ndmin=2).T
        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        output_errors = target - final_output
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot(
            (output_errors * final_output * (1.0 - final_output)),
            numpy.transpose(hidden_output),
        )
        self.wih += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_output * (1.0 - hidden_output)),
            numpy.transpose(input),
        )

    def query(self, inputs):
        """
        Make prediction on input

        Args:
            inputs (numpy.ndarray): Input features

        Returns:
            (numpy.ndarray): Predicted class label
        """
        input = np.array(inputs, ndmin=2).T
        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)
        return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Train and make predict single digits from images"
    )
    parser.add_argument(
        "--hidden-nodes", type=int, default=200, help="Number of nodes in hidden layer"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.2,
        help="Learning rate of neural network",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to run",
    )
    args = parser.parse_args()

    input_nodes = 784
    hidden_nodes = args.hidden_nodes
    output_nodes = 10
    learning_rate = args.learning_rate

    n = neuralNet(input_nodes, hidden_nodes, learning_rate)
    root = Tk()
    root.withdraw()
    test_folder_path = askdirectory(
        title="Select a folder containing test images", parent=root
    )
    root.update()

    data_file = open("mnist_train.csv", "r")
    data_list = data_file.readlines()
    data_file.close()

    epochs = args.epochs
    for e in range(epochs):
        for record in data_list:
            all_values = record.split(",")
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass

    test_data_file = open("mnist_test_10.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    for record in test_data_list:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print(label, "network's answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass

        pass


if __name__ == "__main__":
    main()
