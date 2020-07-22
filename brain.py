import numpy as np
from parameters import brain_inputs, brain_outputs, brain_hnodes


class Brain:
    def __init__(self):
        self.n_inputs = brain_inputs
        self.n_outputs = brain_outputs
        self.n_hnodes = brain_hnodes  # Number of nodes in hidden layer

        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

    def get_inputs(self, state_vec):  # Get inputs from state-vector
        """
        Assign inputs from rover sensors to the input layer of the NN
        :param state_vec: Inputs from rover sensors
        :return: None
        """

        for i in range(self.n_inputs):
            self.input_layer[i, 0] = state_vec[i]

    def get_weights(self, nn_weights):  # Get weights from CCEA population
        """
        Receive NN weights from CCEA or EA
        :param nn_weights:
        :return: None
        """

        self.weights["Layer1"] = np.reshape(np.mat(nn_weights["L1"]), [self.n_hnodes, self.n_inputs])  # 10x8
        self.weights["Layer2"] = np.reshape(np.mat(nn_weights["L2"]), [self.n_outputs, self.n_hnodes])  # 2x10
        self.weights["input_bias"] = np.reshape(np.mat(nn_weights["b1"]), [self.n_hnodes, 1])  # 10x1
        self.weights["hidden_bias"] = np.reshape(np.mat(nn_weights["b2"]), [self.n_outputs, 1])  # 2x1

    def get_outputs(self):
        """
        Run NN to generate outputs
        :return: None
        """
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        for i in range(self.n_hnodes):
            self.hidden_layer[i, 0] = self.sigmoid(self.hidden_layer[i, 0])

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        for i in range(self.n_outputs):
            self.output_layer[i, 0] = self.sigmoid(self.output_layer[i, 0])

    def tanh(self, inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """

        tanh = (2 / (1 + np.exp(-2 * inp))) - 1
        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: sigmoid value
        """

        sig = 1 / (1 + np.exp(-inp))
        return sig

    def run_neural_network(self, state_vec, weight_vec):
        """
        Run through NN for given rover
        :param rover_input: Inputs from rover sensors
        :param weight_vec:  Weights from CCEA
        :return: None
        """
        self.get_inputs(state_vec)
        self.get_weights(weight_vec)
        self.get_outputs()
