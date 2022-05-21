from sklearn.model_selection import train_test_split

from Drebin.deeplearning.deep_neural_network import *
from Drebin.deeplearning.load_data import read

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# np.random.seed(1)

"""
    Deep Learning methodology
    1. Initialize parameters / Define hyperparameters
    2. Loop for num_iterations:
        a. Forward propagation
        b. Compute cost function
        c. Backward propagation
        d. Update parameters (using parameters, and grads from backprop) 
    3. Use trained parameters to predict labels
"""

# model's structure is: [LINEAR -> RELU] * (1) -> LINEAR -> SIGMOID
class two_layer_model:
    num_iterations = None
    learning_rate = None
    parameters = {}
    X = None
    Y = None
    m = None
    grads = {}
    costs = []  # to keep track of the cost
    print_cost = None

    def __init__(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        """
            Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

            Arguments:
            X -- input data, of shape (n_x, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            layers_dims -- dimensions of the layers (n_x, n_h, n_y)
            num_iterations -- number of iterations of the optimization loop
            learning_rate -- learning rate of the gradient descent update rule
            print_cost -- If set to True, this will print the cost every 100 iterations

            parameters -- a dictionary containing W1, W2, b1, and b2
        """
        self.print_cost = print_cost
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.m = self.X.shape[1]  # number of examples
        (n_x, n_h, n_y) = layers_dims

        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        self.parameters = initialize_parameters(n_x, n_h, n_y)


    def train_model(self):
        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        # Loop (gradient descent)

        for i in range(0, self.num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
            A1, cache1 = linear_activation_forward(self.X, W1, b1, activation="relu")
            A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

            # Compute cost
            cost = compute_cost(A2, self.Y)

            # Initializing backward propagation
            dA2 = - (np.divide(self.Y, A2) - np.divide(1 - self.Y, 1 - A2)) # derivative of cost with respect to A2

            # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            self.grads['dW1'] = dW1
            self.grads['db1'] = db1
            self.grads['dW2'] = dW2
            self.grads['db2'] = db2

            # Update parameters.
            self.parameters = update_parameters(self.parameters, self.grads, self.learning_rate)

            # Retrieve W1, b1, W2, b2 from parameters
            W1 = self.parameters["W1"]
            b1 = self.parameters["b1"]
            W2 = self.parameters["W2"]
            b2 = self.parameters["b2"]

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if self.print_cost and i % 100 == 0:
                self.costs.append(cost)

        return self.parameters

if __name__ == "__main__":
    # dataset preparing
    x_all, y_all = read()
    # print(type(y_all), y_all.shape, y_all)
    x_train, x_test, y_train, y_test = train_test_split(x_all.T, y_all.T, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

    # define hyperparameters
    layers_dims = (8, 5, 1)
    learning_rate = 0.03
    num_iterations = 20000
    print_cost = True

    # initial a 2_layers_deep_neural_network model
    dnn_2layers = two_layer_model(x_train, y_train, layers_dims, learning_rate, num_iterations, print_cost)
    parameters = dnn_2layers.train_model()

    pred_train = predict(x_train, y_train, parameters)
    pred_test = predict(x_test, y_test, parameters)

    # plot the cost
    plt.plot(np.squeeze(dnn_2layers.costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

