import numpy as np
from matplotlib import pyplot as plt
import ActivationFunction as AF


# 항등 함수
def identity_function(x):
    return x


# 소프트맥스 함수
def softmax(x):
    if x.ndim == 2:     # 배치 상황일 경우 오버플로우 방지 방법
        x = x.T
        c = np.max(x, axis=0)
        exp = np.exp(x - c)
        y = exp / np.sum(exp, axis=0)
        return y.T
    # 로그 성질을 이용해 자연상수 e의 지수에 C를 빼도 같은 결과가 나옴
    c = np.max(x)  # 오버플로우 방지를 위해 상수 C를 설정
    exp = np.exp(x - c)
    sum_exp = np.sum(exp)
    y = exp / sum_exp
    return y

# def softmax(x):
#     if x.ndim == 2:
#         x = x.T
#         x = x - np.max(x, axis=0)
#         y = np.exp(x) / np.sum(np.exp(x), axis=0)
#         return y.T
#
#     x = x - np.max(x)
#     return np.exp(x) / np.sum(np.exp(x))


# 3층 신경망 구현(입력층 : 3개, 1층 : 4개, 2층: 3개, 출력층 : 2개)===================================================================

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]])
    network['b1'] = np.array([0.2, 0.3, 0.5, 0.7])
    network['W2'] = np.array([[0.3, 0.2, 0.1], [0.2, 0.1, 0.5], [0.5, 0.2, 0.3], [0.3, 0.2, 0.1]])
    network['b2'] = np.array([0.2, 0.2, 0.2])
    network['W3'] = np.array([[0.2, 0.1], [0.2, 0.1], [0.2, 0.1]])
    network['b3'] = np.array([0.2, 0.2])

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(x, W1) + b1
    Z1 = AF.sigmoid(A1)

    A2 = np.dot(Z1, W2) + b2
    Z2 = AF.sigmoid(A2)

    A3 = np.dot(Z2, W3) + b3
    Y = softmax(A3)

    return Y


# X = np.array([1.0, 0.5, 1.5])
# Y = predict(init_network(), X)
# print(Y)

# te = np.array([[5,5,5,6,5,5],[5,5,5,5,5,5],[5,5,5,7,5,5],[5,5,5,8,5,5],[5,5,5,10,5,5]])
# tt = np.array([1,1,1,1,1,1])

