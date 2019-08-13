import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import time
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist, init_network
from SimpleNeuralNetwork import predict
from TwoLayerNet import TwoLayerNet

def image_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)  # 정규화 전처리 True
        # load_mnist(flatten=True, normalize=True, one_hot_label=False) #정규화 전처리 True
    return (x_train, t_train), (x_test, t_test)

def do(): # 배치처리 x   --> 배치처리 했을때와 평균 0.45초 차이남
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    # wrong_idx = []
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 확률이 가장 높은 인덱스 저장
        # print(p, t[i], i)
        print(t[i])
        if p == t[i]:
            accuracy_cnt += 1
        # else:
        #     wrong_idx.append(i)
    print("Accuracy:  " + str(float(accuracy_cnt) / len(x))) # 정규화 o : 93.52 , 정규화 x : 92.07
    # print(wrong_idx)

    # idx = 92
    # print(t[idx])
    # print(np.argmax(predict(network,x[idx])))
    # img = x[idx].reshape(28, 28)
    # image_show(img)

def do_batch():
    x, t = get_data()
    network = init_network()
    batch_size = 1000
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i: i + batch_size])

    print("Accuracy:  " + str(float(accuracy_cnt) / len(x))) # 정규화 o : 93.52 , 정규화 x : 92.07


# current_milli_time = lambda: int(round(time.time() * 1000))
#
# start = current_milli_time()
# do_batch()
# end = current_milli_time()
# print(end - start)
#
# start = current_milli_time()
# do()
# end = current_milli_time()
# print(end - start)


### ==================================신경망 학습 구현==========================================================
# 학습 데이터 로드
(x_train, t_train), (x_test, t_test) = get_data()

train_loss_list = []
train_acc_list = []
test_acc_list = []


iters_num = 10000   # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_per_epoch = max(train_size / batch_size, 1)

# 학습 시작
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)   # 어레이 인덱스를 범위 내에서 무작위로 생성
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print(i)
    # grad = network.numerical_gradient(x_batch, t_batch)     # 수치미분을 사용해 신경망의 손실함수에 대한 weight, bias 미분값을 구함
    grad = network.gradient(x_batch, t_batch)               # 오차역전파를 사용해 매개변수 구함
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]    # 미분값을 학습률과 곱하여 기존값에서 빼는 방식으로 갱신


    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))



