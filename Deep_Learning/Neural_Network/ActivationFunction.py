import numpy as np
from matplotlib import pyplot as plt

# 계단 함수 구현
# 인수 -> 실수(부동소수점)만 가능
def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0

# 계단 함수 넘파이 배열 인수
def step_function_2(x):
    y = x > 0
    return y.astype(np.int)

# x = np.array([-1.0, 1.0, 2.0])
# print(x)
# y = x > 0
# print(y)
# print(y.astype(np.int))

def step_function(x):
    return np.array(x > 0,  dtype=np.int)

# x = np.arange(-5.0, 5.0, 0.1) # -5.0 에서 5.0 까지 0.1 간격의 넘파이 배열 생성 // 죽 [-5.0, -4.9, -4.8, -4.7 ..... 4.8, 4.9]을 생성
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # y축 범위 지정
# plt.show()

# 시그모이드 함수 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 넘파이 배열이 넘어와도 처리되도록 구현

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()


# ReLU 함수
def relu(x):
    return np.maximum(0, x) # maximum은 두 입력 중 큰 값을 선택해 반환

# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# a = np.array([[1,2,3],[4,5,6]])
# b = np.array([1,2])
# print(np.dot(b,a))






