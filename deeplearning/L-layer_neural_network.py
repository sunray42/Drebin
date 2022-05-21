from sklearn.model_selection import train_test_split

from Drebin.deeplearning.deep_neural_network import *
from Drebin.deeplearning.load_data import undersampling

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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

# model's structure is: [LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID
class L_layer_model:
    num_iterations = None
    learning_rate = None
    parameters = {}
    X = None
    Y = None
    m = None
    grads = {}
    costs = []  # to keep track of cost
    print_cost = None

    def __init__(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
        """
        self.print_cost = print_cost
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.m = self.X.shape[1]  # number of examples

        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        self.parameters = initialize_parameters_deep(layers_dims)

    def train_model(self):
        for i in range(0, self.num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(self.X, self.parameters)

            # Compute cost.
            cost = compute_cost(AL, self.Y)

            # Backward propagation.
            self.grads = L_model_backward(AL, self.Y, caches)

            # Update parameters.
            self.parameters = update_parameters(self.parameters, self.grads, self.learning_rate)

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                self.costs.append(cost)

        return self.parameters

if __name__ == "__main__":
    # dataset preparing
    x_all, y_all = undersampling(neg_pos_ratio=1)
    # print(type(y_all), y_all.shape, y_all)
    x_train, x_test, y_train, y_test = train_test_split(x_all.T, y_all.T, test_size=0.3, random_state=42)
    x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

    # define hyperparameters
    layers_dims = (8, 16, 8, 4, 1)
    learning_rate = 0.1
    num_iterations = 20000
    print_cost = True

    # initial a 2_layers_deep_neural_network model
    dnn_L_layers = L_layer_model(x_train, y_train, layers_dims, learning_rate, num_iterations, print_cost)
    parameters = dnn_L_layers.train_model()

    print()
    print("Training set:")
    pred_train = predict(x_train, y_train, parameters)
    print()
    print("Testing set:")
    pred_test = predict(x_test, y_test, parameters)

    # plot the cost
    plt.plot(np.squeeze(dnn_L_layers.costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()