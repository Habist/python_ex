import sys, os
sys.path.append(os.pardir)
import numpy as np
from SimpleNeuralNetwork import softmax
from ActivationFunction import sigmoid
from LossFunction import cross_entropy_error_batch,cross_entropy_error
from Differentiation import numerical_gradient
from Layer import *
from collections import OrderedDict


class MultiLayerNet:
    def __init__(self, input_size, output_size,
                 hidden_size_list = [100, 100, 100], use_dropout = False):
        # 지역변수 초기화
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_length = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.size_list = []
        self.layers = OrderedDict()
        self.params = {}

        # size_list 초기화
        self.size_list.append(self.input_size)
        self.size_list.extend(self.hidden_size_list)
        self.size_list.append(self.output_size)
        self.size_list_length = len(self.size_list)

        # 계층 및 매개변수 생성
        for idx in range(self.size_list_length):
            if idx != self.size_list_length - 1:    # 마지막 계층을 제외하고 생성
                W = np.random.randn(self.size_list[idx], self.size_list[idx+1]) / np.sqrt(self.size_list[idx])
                b = np.zeros(self.size_list[idx+1])

                self.params['W' + str(idx+1)] = W
                self.params['b' + str(idx+1)] = b
                self.layers['Affine' + str(idx+1)] = Affine(W, b)

                if idx != self.size_list_length - 2:    # hidden_layer의 마지막 활성화 함수는 생성할 필요 없음
                    self.layers['Relu' + str(idx+1)] = Relu()
            else:
                self.lastLayer = SoftmaxWithLoss() # 마지막 계층 생성


    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

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

        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_W, self.params[key])

        # grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

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

        for key in self.layers.keys():
            idx = key[-1:]
            if key[:-1] == "Affine":    # Affine 신경망의 매개변수 저장
                grads['W' + idx] = self.layers[key].dW
                grads['b' + idx] = self.layers[key].db

        # grads['W1'] = self.layers['Affine1'].dW
        # grads['b1'] = self.layers['Affine1'].db
        # grads['W2'] = self.layers['Affine2'].dW
        # grads['b2'] = self.layers['Affine2'].db

        return grads


# MultiLayerNet(784, 10, hidden_size_list=[200, 100, 50, 100])