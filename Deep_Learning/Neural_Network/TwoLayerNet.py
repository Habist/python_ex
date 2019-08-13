import sys, os
sys.path.append(os.pardir)
import numpy as np
from SimpleNeuralNetwork import softmax
from ActivationFunction import sigmoid
from LossFunction import cross_entropy_error_batch,cross_entropy_error
from Differentiation import numerical_gradient
from Layer import *
from collections import OrderedDict

# 스탠포드 대학교의 CS231n 수업에서 제공한 소스 코드를 참고함
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):

        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()         # 순서가 있는 딕셔너리 // 추가한 순서대로 들어감
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()


    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

        # =====Layer Class 미사용 방법===========
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        #
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        #
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)
        #
        # return y

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

        # return cross_entropy_error_batch(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        # t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy * 100

    # 수치 미분을 사용한 기울기 구하기
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 역전파를 이용한 기울기 구하기
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1        # 역전파의 시작 값은 1로 시작 ( Why ?  출력값에 해당하는 국소적 미분값이 1이기 때문)
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error_batch(y, t)

        return loss


# net = SimpleNet()
# x = np.array([0.5, 0.8])
# p = net.predict(x)
# t = np.array([0,0,1])

# def f(W): # 람다로 표현하면 f = lambda w: net.loss(x, t)
#     return net.loss(x, t)

# print(numerical_gradient(lambda w: net.loss(x, t), net.W))