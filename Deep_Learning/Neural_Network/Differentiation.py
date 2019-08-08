import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist, init_network


# 수치 미분 나쁜 구현 예
def wrong_numerical_diff(f, x):
    h = 10e-50  # lim h->0 와 가장 가깝게 구현하는 방법
    # 하지만 반올림 오차 문제를 발생시킴(32비트 부동소수점에서 너무 작은 값은 컴퓨터가 계산하는 데 문제가 발생하므로
    # 이것을 개선해야 함
    return (f(x + h) - f(x)) / h


# 수치 미분 좋은 구현 예
def numerical_diff(f, x):
    h = 1e-4  # 0.0001 적당히 작은 값을 줘서 반올림 오차 문제를 회피
    return (f(x + h) - f(x - h)) / (2 * h)  # 전진, 후진 차분보다 중앙 차분이 오차가 낮기 때문에 사용한다.


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)


def linear(a, b, x):
    return a * x + b


# x에서의 접선 구하기
def make_gradient(f, x, x_range):
    a = numerical_diff(f, x)  # 기울기
    b = f(x) - a * x  # 상수

    y = (lambda t: a * t + b)  # x에서의 접선 함수 저장
    # linear(a,b,x_range)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x_range, f(x_range))  # f함수 그리기
    plt.plot(x_range, y(x_range))  # 접선 함수 그리기
    plt.show()


def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]        # 미분을 하는 x값만 차분을 이용하기 위해 임시 저장
        x[idx] = tmp_val + h    # 미분을 위해 기존의 값에 + h 후 다시 저장
        fxh1 = f(x)             # 현재 인덱스의 x값에서 미분을 하기 때문에 차분할 값이 함께 있는 x 어레이를 인자로 넣음

        x[idx] = tmp_val - h    # 똑같이 차분을 위한 과정
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)    # 중앙차분 공식
        x[idx] = tmp_val                     # 기존 x값 원복

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x      # 시작 x값 설정

    for i in range(step_num):           # step_num 만큼 학습 진행
        grad = numerical_gradient(f,x)  # x에서의 기울기를 구하고
        x -= lr * grad                  # 기존 x를 학습률과 기울기를 곱한 값을 뺀다. 그러면 기울기가 0으로 수렴

    return x                        # 결국 함수의 최솟값으로 간다.(함수의 출력값이 가장 작은 곳이 최솟값 부근이기 때문)


print(gradient_descent(function_2, np.array([-3.0, 4.0]), lr=0.1, step_num=100))


# make_gradient(function_1, 20, np.arange(0.0, 20.0, 0.1)) # function_1 함수의 x = 20인 값에서 접선을 그리기
# x0, x1 = np.arange(-3,3,1)
# y = function_2([x0, x1])
# plt.plot([x0, x1] ,y)
# plt.show()



