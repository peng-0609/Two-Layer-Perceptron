# Author Kun Peng
# Command to run the program:
# python Lab1.py
import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_sig(x):
    return x * (1 - x)


def neuron_layer(num_of_input, num_of_neuron):
    # initializing all weights of floats in range [-1,1)
    return np.random.random((num_of_input, num_of_neuron)) * 2 - 1


def output(layer1, layer2, input):
    # setting bias to 1's
    bias_layer2 = np.ones(len(input)).transpose()
    input_layer2 = np.c_[sigmoid(np.dot(input, layer1)), bias_layer2]
    return np.array([sigmoid(np.dot(input_layer2, layer2))])


def backprop(x, y, eta, alpha):
    # iterate every learning rate
    for i in range(len(eta)):
        start_time = time.time()
        np.random.seed(30)
        # hidden layer with 4 neurons, 5 inputs per neuron from x1-x4 and b
        layer1 = neuron_layer(5, 4)
        # second layer with 1 neuron and 5 inputs from layer1 and b
        layer2 = neuron_layer(5, 1)
        iterate = True
        epoch = 0
        error = []
        while iterate is True:

            epoch += 1
            # arrays for delta weights from previous iteration
            dw2_prev = np.zeros((np.shape(layer2)[0], np.shape(layer2)[1]))
            dw1_prev = np.zeros((np.shape(layer1)[0], np.shape(layer1)[1]))
            # iterate every input of x
            for j in range(len(x)):
                phi_layer1 = np.array([sigmoid(np.dot(x[j], layer1))])
                # append bias of layer 2
                input_layer2 = np.append(phi_layer1, 1.0)
                phi_layer2 = sigmoid(np.dot(input_layer2, layer2))
                # delta rule
                # d_layer2 = e * diff_sig(phi)
                d_layer2 = (np.array([y[j]]) - phi_layer2) * diff_sig(phi_layer2)
                # layer2[:-1, :]) refers to all weights of inputs from neurons from layer1(ignoring w of bias)
                d_layer1 = diff_sig(phi_layer1) * np.transpose(layer2[:-1, :]) * d_layer2
                # delta w with momentum (alpha * dw_prev)
                d_w2m = np.transpose(eta[i] * d_layer2 * input_layer2) + np.multiply(alpha, dw2_prev)
                d_w1m = eta[i] * np.array([x[j]]).transpose() * d_layer1 + np.multiply(alpha, dw1_prev)
                # update weights and delta w(n-1)
                layer2 += d_w2m
                layer1 += d_w1m
                dw2_prev = d_w2m
                dw1_prev = d_w1m

            out = output(layer1, layer2, x)
            error = np.abs(y - out)
            # stop learning if error < 0.05
            if np.max(error) < 0.05:
                iterate = False
            # if epoch % 100000 == 0:
            #     print('For eta = {:.2f}, epoch {}, max error is {}'.format(eta[i], epoch, np.max(error)))
        print('For eta = {:.2f}, epoch is {}, training time is {:.2f} seconds'.format(eta[i], epoch,
                                                                                      (time.time() - start_time)))


if __name__ == "__main__":
    eta = np.arange(0.05, 0.55, 0.05)
    # initialize input array with all possible inputs and random bias
    x = np.array(([0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1], [1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [1, 1, 1, 0, 1],
                  [1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]), dtype=int)
    y = np.array(([0], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [0]), dtype=int)
    print('Training without momentum:')
    backprop(x, y, eta, 0)
    print('\n\nTraining with alpha = 0.9:')
    backprop(x, y, eta, 0.9)
