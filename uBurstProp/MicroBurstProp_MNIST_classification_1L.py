# Building Microburst prop from scratch
# Inspired by https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import random as rd

np.random.seed(42)

# Defining details of the model
input_size = 28 * 28    # img_size = (28,28) ---> 28*28=784 in total
num_classes = 10        # number of output classes discrete range [0,9]
lr = 0.05               # Learning rate: size of step
repeat_count = 100      # Repeating each class for a better noise attenuation
hidden_n = 0        # Nodes in the hidden layer
output_n = repeat_count * num_classes

num_epochs = 20       # number of times which the entire dataset is passed throughout the model
num_images = 100       # Number of images in each batch for updating P

# ------------------------------------------------------------------------------
      ##               Importing data (MNIST using Keras)
# -------------------------------------------------------------------------------

# for importing data in numpy format
from keras.datasets import mnist

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
        return X_train.T, y_train.T, X_val.T, y_val.T, X_test.T, y_test.T


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

# pause the program if this condition does not meet
# Make sure data is 784 (28 x 28) rows
assert X_train.shape[0] == input_size, "Input size is not correct!"

# --------------------------------------------------------------------------------- #
##                    Define the classes                  ##
# --------------------------------------------------------------------------------- #

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    # A building block - features of each layer are:
    # 1. process input to estimate the output (in forward pass) // output = layer.forward(input)
    # 2. propagate gradients backward // grad_input = layer.backward(input, grad_output)
    # 3. In some layers, we have learnable parameters which get updated in backward pass

    def __init__(self):
        # Here the layer parameters can be initialized
        # A dummy layer does not have a parameter
        pass

    def forward(self, input):
        # Take input data of shape [batch, input units(or features)] and returns output data [batch, output units]
        # A dummy laryer only return the input:
        return input

    def backward(self, input, grad_output):
        # Performs back propagation step
        # For estimating the loss gradient, we apply the chain rule: d loss / dx = (d loss / d layer) * (d layer / dx)
        # Here we have (d loss/d layer) as grad_output, so we only need to multiply it by d layer / dx
        # The learnable parameters (if any -- depending on layer type) should also get updated in this function
        # The gradient of a dummy layer is grad_output itself, but we write it in a better way here:
        n_units = input.shape[1]
        d_layer__d_input = np.eye(n_units)  # because it is a dummy layer
        return np.dot(grad_output, d_layer__d_input)  # chain rule


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        return sigmoid_function(input)

    def backward(self, input, grad_output):
        n_units = input.shape[1]
        sigmoid_grad = sigmoid_function(input) * (1 - sigmoid_function(input))
        return grad_output * sigmoid_grad


class Dense_uBurstProp(Layer):
    def __init__(self, input_units, output_units, next_layer_units=0, learning_rate=0.1):

        # A dense layer which performs Burst propagation rather than backProp
        self.learning_rate = learning_rate

        # weights initialized with Xavier initialization approach
        self.W = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_units + output_units)),
                                  size=(output_units, input_units))  # forward weights
        self.b = np.zeros(output_units)  # forward biases

        if next_layer_units == 0:  # If this is the last layer of the network
            self.Y = []
            self.c = []
        else:
            self.Y = np.random.normal(loc=0.0, scale=np.sqrt(2 / (output_units + next_layer_units)),
                                      size=(output_units, next_layer_units))  # forward weights
            self.c = np.zeros(output_units)  # forward biases

        # event, burst, burst probability variables, eligibility traces, burst counts, and event count
        self.E = np.zeros(output_units)
        self.B = np.zeros(output_units)
        self.P = np.zeros(output_units) + 0.2  # Based burstprop paper initial values suggestion
        self.G = np.zeros(input_units)
        self.BK = np.zeros(output_units)
        self.EK = np.zeros(output_units) + 0.001  # To avoid multiplication by 0

    def forward(self, input):

        if len(np.shape(input)) == 2:
            batch_length = input.shape[1]
        else:
            batch_length = 1

        # Get exposed to one image in the training phase
        if batch_length == 1:
            # Determine the probability of an event in each unit
            P_E = sigmoid_function(np.dot(self.W, input)) + self.b

            # Sample the events based on the event probabilities
            self.E = 1.0 * (np.random.random_sample(self.E.shape) < P_E)

            # update the event counts
            self.EK += self.E

            # Store the eligibility
            self.G = input

            return self.E

        # Get exposed to more than one image in the prediction phase
        else:
            # Determine the probability of an event in each unit
            P_E = sigmoid_function(np.dot(self.W, input)) + np.squeeze(
                np.broadcast_to(self.b.reshape(len(self.b), 1), (len(self.b), batch_length)))

            # Sample the events based on the event probabilities
            predicted_E = 1.0 * (np.random.random_sample(P_E.shape) < P_E)

            return predicted_E

    def backward(self, feedback, target=-1, BP_update=False):

        # Weight decay hyper-parameter
        lambda_param = 0.00001

        # Check whether we should update the burst probability
        if BP_update:
            self.P = self.BK / self.EK

        # Check whether this is an output layer with a target
        if target is -1:

            # Calculate the burst probabilites
            P_B = sigmoid_function(np.dot(self.Y, feedback) + self.c) * self.E

            # Sample the bursts
            self.B = 1.0 * (np.random.random_sample(self.B.shape) < P_B)

        else:

            # Set the bursts based on the target
            self.B = np.squeeze(target)

        # update the burst counts
        self.BK += self.B

        # Update the weights and biases
        delta_W = np.outer((self.B - self.P * self.E), self.G)
        delta_b = self.B - self.P * self.E
        self.W += (self.learning_rate * delta_W - lambda_param * self.W)
        self.b += self.learning_rate * delta_b

        #print([delta_W.max(), delta_W.min(), self.W.max(), self.W.min()])

        return self.B


# Define the network and do the loops
learning_rate = lr

# ------------------------------------------------------------
               # Building the nework
# -------------------------------------------------------------
network = []
network.append(Dense_uBurstProp(X_train.shape[0], output_n, 0, learning_rate))


def forward(network, X):
    # Set the input
    input = X

    # Looping through each layer
    for l in network:
        # Calculate the event rates
        input = l.forward(input)

    # return the activation of last layer (output layer)
    return input


def backward(network, targets, BP_update):
    # Set the output flag
    output = True

    # Run backwards through the network

    for l in reversed(network):
        # for l_index in range(len(network))[::-1]:
        #  l = network[l_index]

        # Calculate the bursts and update the weights
        if output:
            feedback = l.backward(0, targets, BP_update)
            output = False
        else:
            feedback = l.backward(feedback)


def predict(network, X):
    # Determine which of the 100 unit pieces had the largest number of active units
    output = forward(network, X)

    # Extract the frequency of each digit getting selected in the output layer
    n_batch = output.shape[1]
    summed_output = output.reshape(num_classes, -1, n_batch).sum(1)
    assert summed_output.shape[0] == num_classes, "Not correct number of classes in the prediction function"

    # Return the selected class: digit with more "ones" for it in the output layer

    # Because there could be more than one class with the exact same number of
    # ones (even with no one in total --> zero will be selected), we compare the
    # digit with first and second larger number of ones and return -1 if there
    # are more than one digit with same number of ones, we set it as not classified --> -1
    twomax = np.partition(summed_output, -2, axis=0)[-2:, :].T
    default = -1
    selected_digit = np.where(twomax[:, 0] != twomax[:, 1], summed_output.argmax(axis=0), default)

    return selected_digit


def train(network, X, y, BP_update):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.

    # run the forward pass
    output = forward(network, X)  # repeated one output

    # Set the targets based on batch size

    if isinstance(y, (list, tuple, np.ndarray)):
        y_one_transform = np.zeros([y.shape[0], num_classes])
        for i, e in enumerate(y):
            y_one_transform[i, e] = 1

    else:
        y_one_transform = np.zeros(num_classes)
        y_one_transform[y] = 1

    targets = np.squeeze(np.tile(y_one_transform, [1, repeat_count]).T)  # We have "repeat_count" times of output classes

    # Run the backward pass
    # Propagate burst rates through the network with Reverse propogation
    backward(network, targets, BP_update)

    # Calculate error rate
    assert targets.shape == output.shape, "The size of output of the network does not match the repeated ones target"
#    error_rate = (np.square(output - targets)).mean()
    error_rate = np.mean(output == targets)

    return error_rate.mean()

# -----------------------------------------------------------------------------#
##                Training the network                                        ##
# -----------------------------------------------------------------------------#
from tqdm import trange
from IPython.display import clear_output


train_log = []
val_log = []
ER_rate = []
num_batch = int(len(X_train.T)/num_images)

for epoch in range(num_epochs):

    # Randomize the order of images in the data set for training
    indices = np.random.permutation(len(X_train.T))
    #indices_subsample = indices[0:num_images]
    x_rand = X_train[:,indices]
    y_rand = y_train[indices]

    for batch in range(num_batch):
        # set the BP update flag
        # Burst probability updating at the end of each epoch (or at the beginning of the next epoch)
        BP_update = True

        # In each epoch use a new group of images (instead of randomizing)
        x_batch = x_rand[:, batch * num_images: (batch + 1) * num_images]
        y_batch = y_rand[batch * num_images: (batch + 1) * num_images]

        # Training on single image at a time
        for i_image in range(num_images):
            ER_rate.append(train(network, x_batch[:, i_image], y_batch[i_image], BP_update))
            if BP_update:
                BP_update = False

    # Calculate error rate
    train_log.append(np.mean(predict(network, X_train) == y_train))
    val_log.append(np.mean(predict(network, X_val) == y_val))

    # Printing and illustrating the error
    clear_output()
    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Val accuracy:", val_log[-1])
    print("Training Error:", ER_rate[-1])

    plt.subplot(1, 2, 1)
    plt.plot(train_log, label='train accuracy')
    plt.plot(val_log, label='val accuracy')
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(ER_rate, label='Training Error')
    plt.legend(loc='best')
    plt.grid()
    # plt.show()
    # Saving the figure
    plt.savefig('Accuracy_Error_uBurstProp_1L_test3_lr001_RC200.png')


# Saving the trained network
import yaml
from pathlib import Path

network_dictionary = dict(N1 = network[0].__dict__, N2 = network[1].__dict__)
performance_dictionary = dict(Training_acc = train_log , Val_acc = val_log, TrainingEr = ER_rate)
#dest = Path("python_projects/uBurstProp")
with open(r'network_model_1L.yml', 'w') as outfile:
    yaml.dump(network_dictionary, outfile, default_flow_style=False)

with open(r'result_perf_1L.yml', 'w') as outfile:
    yaml.dump(performance_dictionary, outfile, default_flow_style=False)


# Reading process
#with open(r'E:\data\fruits.yaml') as file:
#    # The FullLoader parameter handles the conversion from YAML
#    # scalar values to Python the dictionary format
#    fruits_list = yaml.load(file, Loader=yaml.FullLoader)
#
#    print(fruits_list)