import numpy as np
import pandas as pd


class train_Neural_Network:
    def __init__(self, input_size, output_size, hidden_size, no_of_examples, input_matrix, output_matrix, test_X, test_size):
        self.input_size = input_size  # 3x1
        self.output_size = output_size  # 1x1
        self.hidden_size = hidden_size  # 2x1
        self.no_of_examples = no_of_examples  # 3x1
        self.a1 = np.zeros((self.hidden_size, 1))  # 2x1
        self.a2 = np.zeros((self.output_size, 1))  # 1x1
        self.w1 = np.random.randn(self.input_size, self.hidden_size)  # 3x2
        self.b1 = np.random.randn(self.hidden_size, 1)  # 2x1
        self.w2 = np.random.randn(self.hidden_size, self.output_size)  # 2x1
        self.b2 = np.random.randn(self.output_size, 1)  # 1x1
        self.dloss1 = np.zeros((self.hidden_size, self.input_size))  # 2x3
        self.dloss2 = np.zeros((self.output_size, self.hidden_size))  # 1x2
        self.X = input_matrix  # 3x3
        self.y = output_matrix  # 3x1
        self.test_X = test_X
        self.test_size = test_size
        self.index = 0
        self.connections()

    def extracting_input_vector(self):
        self.input_vector = self.X[self.index, :].reshape(-1, 1)  # 3x1

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fwd_prop(self):
        z1 = np.dot(self.w1.T, self.input_vector) + self.b1  # 2x1
        self.a1 = self.sigmoid(z1)  # 2x1
        z2 = np.dot(self.w2.T, self.a1) + self.b2  # 1x1
        self.a2 = self.sigmoid(z2)  # 1x1

    def transform_y(self):
        self.transformed_y = np.zeros((self.output_size, 1))
        self.transformed_y[self.y[self.index]] = 1

    def bwrd_prop(self, alpha=0.01):
        m = self.no_of_examples
        self.dloss2 += np.dot((self.a2-self.transformed_y), self.a1.T)  # 1x2
        self.dloss1 += np.dot((self.w2*self.a1*(1-self.a1)),
                              np.dot(self.a2-self.transformed_y, self.input_vector.T))  # 2x3
        if (self.index <= self.no_of_examples-1):
            self.connections()
        else:
            self.w2_dash = self.w2.T - (alpha/m)*self.dloss2
            self.w1_dash = self.w1.T - (alpha/m)*self.dloss1
            self.w1 = self.w1_dash.T
            self.w2 = self.w2_dash.T
            self.display_output()

    def connections(self):
        if (self.index <= self.no_of_examples-1):
            self.extracting_input_vector()
            self.fwd_prop()
            self.transform_y()
            self.index += 1
            self.bwrd_prop()

    def maximum_value_index(self):
        maximum_value = 0
        self.maximum_index = -1
        for i in range(self.output_size):
            if (self.a2[i] >= maximum_value):
                maximum_value = self.a2[i]
                self.maximum_index = i

    def display_output(self):
        self.input_matrix = self.test_X
        self.index = 0
        self.no_of_examples = self.test_size
        while (self.index < self.no_of_examples):
            self.extracting_input_vector()
            self.fwd_prop()
            self.maximum_value_index()
            print(self.maximum_index)
            self.index += 1
