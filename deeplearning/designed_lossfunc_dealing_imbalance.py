from sklearn.model_selection import train_test_split

from Drebin.deeplearning.deep_neural_network import *
from Drebin.deeplearning.load_data import undersampling, oversampling

import numpy as np
import matplotlib.pyplot as plt
from deeplearning.focal_loss import binary_focal_loss

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
    costs = []
    def __init__(self, X, Y, layers_dims, optimizer='gd',
                 beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
                 learning_rate=0.0075, num_epochs=3000, print_cost=False,
                 loss_type = 'cross_entropy',  alpha=None,
                 lambd=0.0, keep_prob=1.0, mini_batch_size=128):
        """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_epochs -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps
            lambd -- regularization hyperparameter, scalar
            keep_prob -- probability of keeping a neuron active during drop-out, scalar.
            mini_batch_size -- size of the mini-batches, integer
            beta -- Momentum hyperparameter
            beta1 -- Exponential decay hyperparameter for the past gradients estimates
            beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
            epsilon -- hyperparameter preventing division by zero in Adam updates

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
        """
        self.optimizer = optimizer
        self.print_cost = print_cost
        self.num_epochs = num_epochs
        self.loss_type = loss_type
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.m = self.X.shape[1]  # number of examples
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        self.parameters = initialize_parameters_deep(layers_dims)

        # Initialize the optimizer
        if optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif optimizer == "momentum":
            self.v = initialize_velocity(self.parameters)
        elif optimizer == "adam":
            self.v, self.s = initialize_adam(self.parameters)

    def train_model(self):
        t = 0       # initializing the counter required for Adam update

        for i in range(0, self.num_epochs):
            minibatches = random_mini_batches(self.X, self.Y, self.mini_batch_size, seed=10)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                if self.keep_prob <= 1:
                    AL, caches = L_model_forward(minibatch_X, self.parameters, self.keep_prob)
                else:
                    print('keep_prob cannot be greater than 1')
                    break

                # Compute cost.
                if self.loss_type == 'focal_loss':
                    loss_func = binary_focal_loss()
                    cost = loss_func(minibatch_Y, AL)
                elif self.loss_type == 'cross_entropy':
                    if self.lambd == 0:
                        cost = compute_cost(AL, minibatch_Y)
                    else:
                        cost = compute_cost_with_regularization(AL, minibatch_Y, self.parameters, self.lambd, self.alpha)
                elif self.loss_type == 'weighted_CE':
                    if self.alpha == None:
                        print("alpha should be given to compute weight_CE!")
                        break
                    elif self.lambd == 0:
                        cost = compute_weighted_cross_entropy_loss(AL, minibatch_Y, self.alpha)
                    else:
                        cost = compute_cost_with_regularization(AL, minibatch_Y, self.parameters, self.lambd, self.alpha)


                if self.lambd == 0 and self.alpha == None:
                    cost = compute_cost(AL, minibatch_Y)
                elif self.lambd == 0 and self.alpha is not None:
                    cost = compute_weighted_cross_entropy_loss(AL, minibatch_Y, self.alpha)
                else:
                    cost = compute_cost_with_regularization(AL, minibatch_Y, self.parameters, self.lambd, self.alpha)

                # Backward propagation.
                self.grads = L_model_backward(AL, minibatch_Y, caches, self.lambd, self.keep_prob)

                # Update parameters
                if optimizer == "gd":
                    self.parameters = update_parameters(self.parameters, self.grads, self.learning_rate)
                elif optimizer == "momentum":
                    self.parameters, self.v = update_parameters_with_momentum(self.parameters, self.grads,
                                                                              self.v, self.beta,
                                                                              self.learning_rate)
                elif optimizer == "adam":
                    t = t + 1  # Adam counter
                    self.parameters, self.v, self.s = update_parameters_with_adam(self.parameters, self.grads,
                                                                                  self.v, self.s, t,
                                                                                  self.learning_rate,
                                                                                  self.beta1, self.beta2, self.epsilon)

            # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self.print_cost and i % 100 == 0:
                self.costs.append(cost)

        return self.parameters

if __name__ == "__main__":
    # dataset preparing
    x_all, y_all = oversampling(repeat_sampling_times=8)
    # print(type(y_all), y_all.shape, y_all)
    x_train, x_test, y_train, y_test = train_test_split(x_all.T, y_all.T, test_size=0.3, random_state=42)
    x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

    # define hyperparameters
    layers_dims = (8, 16, 8, 4, 1)
    optimizer = 'momentum'        # optional optimizer: gd / momentum / adam
    beta = 0.9
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    learning_rate = 0.1
    num_epochs = 8800
    print_cost = True
    loss_type = 'cross_entropy'    # loss function category: cross_entropy / weighted_CE / focal_loss
    alpha = None      # weighted_CE hyperparameter, alpha is commonly set to neg_pos_ratio / (neg_pos_ratio+1)
    lambd = 0      # L2 regularization hyperparameter, lambd = 0: no L2 regularization
    keep_prob = 1     # dropout hyperparameter, keep_prob = 1: no dropout
    print("size of training set: ", x_train.shape[1])
    mini_batch_size = 4096 # x_train.shape[1]
    # mini_batch_size == 1, SGD; 1 < mini_batch_size < m mini-batch GD; mini_batch_size == m batch GD

    # initial a 2_layers_deep_neural_network model
    dnn_L_layers = L_layer_model(x_train, y_train, layers_dims, optimizer,
                                 beta, beta1, beta2, epsilon,
                                 learning_rate, num_epochs, print_cost,
                                 loss_type, alpha,
                                 lambd, keep_prob, mini_batch_size)
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