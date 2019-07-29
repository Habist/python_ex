import numpy as np
from matplotlib import pyplot as plt
import ActivationFunction as AF

#항등 함수
def identity_function(x):
    return x

# 3층 신경망 구현(입력층 : 3개, 1층 : 4개, 2층: 3개, 출력층 : 2개)===================================================================

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]])
    network['B1'] = np.array([0.2, 0.3, 0.5, 0.7])
    network['W2'] = np.array([[0.3, 0.2, 0.1],[0.2, 0.1, 0.5],[0.5, 0.2, 0.3],[0.3, 0.2, 0.1]])
    network['B2'] = np.array([0.2, 0.2, 0.2])
    network['W3'] = np.array([[0.2, 0.1], [0.2, 0.1], [0.2, 0.1]])
    network['B3'] = np.array([0.2, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    A1 = np.dot(x, W1) + B1
    Z1 = AF.sigmoid(A1)

    A2 = np.dot(Z1, W2) + B2
    Z2 = AF.sigmoid(A2)

    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)

    return Y

X = np.array([1.0, 0.5, 1.5])
Y = forward(init_network(), X)
print(Y)






