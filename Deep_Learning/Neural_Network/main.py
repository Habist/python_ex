import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import time
import numpy as np
import pickle
from PIL import Image
from dataset.mnist import load_mnist, init_network
from SimpleNeuralNetwork import predict
from TwoLayerNet import TwoLayerNet
from Optimization import *
from MultiLayerNet import MultiLayerNet


def image_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()


def shuffle_dataset(x, t):

    permutation = np.random.permutation(x.shape[0])    # 인자만큼의 인덱스를 무작위로 반환
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)  # 정규화 전처리 True
        # load_mnist(flatten=True, normalize=True, one_hot_label=False) #정규화 전처리 True
    return (x_train, t_train), (x_test, t_test)


def get_data_with_validation():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    x_train, t_train = shuffle_dataset(x_train, t_train)

    # 20%를 검증 데이터로 분할
    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)

    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]

    return (x_train, t_train), (x_test, t_test), (x_val, t_val)


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


def gradient_check():
    (x_train, t_train), (x_test, t_test) = get_data()

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    # 각 가중치의 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 냄
    for key in grad_numerical.keys():
        diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
        print(key + " : " + str(diff))


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

def train():
    ### ==================================신경망 학습 구현==========================================================
    # 학습 데이터 로드
    (x_train, t_train), (x_test, t_test) = get_data()

    # 오버피팅 강제로 재현 // 적은 수의 데이터로 학습을 진행 -> 오버피팅 발생 -> 범용성 떨어짐
    # x_train = x_train[:300]
    # t_train = t_train[:300]

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []


    iters_num = 3000   # 반복 횟수
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    # network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)  # 2계층 신경망
    network = MultiLayerNet(784, 10, hidden_size_list=[100, 100, 50], use_batchNorm=True, use_dropout=True, dropout_ratio=0.15)  # 다계층 신경망 // Hidden layer 동적 할당

    # optimizer = SGD(lr=0.01)   # 확률적 경사 하강법(SGD) Default Learning rate = 0.01
    # optimizer = Momentum()    # 모멘텀 // 기울기 방향으로 힘을 받아 물체가 가속된다는 물리 법칙
    # optimizer = AdaGrad()     # AdaGrad // 과거의 기울기를 제곱하여 더해감 -> 갱신 강도 감소
    optimizer = Adam()        # Adam // 모멘토와 AdaGrad를 융합한듯한 방법 원논문 참고

    iter_per_epoch = max(train_size / batch_size, 1)



    # 학습 시작
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)   # 어레이 인덱스를 범위 내에서 무작위로 생성
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # grads = network.numerical_gradient(x_batch, t_batch)     # 수치미분을 사용해 신경망의 손실함수에 대한 weight, bias 미분값을 구함
        grads = network.gradient(x_batch, t_batch)               # 오차역전파를 사용해 매개변수 구함
        optimizer.update(network.params, grads)                  # 매개변수 갱신

        # for key in ('W1', 'b1', 'W2', 'b2'):
        #     network.params[key] -= learning_rate * grad[key]    # 미분값을 학습률과 곱하여 기존값에서 빼는 방식으로 갱신

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # 하나의 값 예측
    print(network.accuracy(x_test[[2]], t_test[[2]]))

    #학습 객체 저장
    with open("./saveNetwork/FourLayersNetwork.pkl","wb") as file:
        pickle.dump(network, file)
        file.close()

    return network

# 저장한 객체 불러오기
def get_network():
    with open("./saveNetwork/FourLayersNetwork.pkl","rb") as file:
        network = pickle.load(file)

        return network





network = train()

(x_train, t_train), (x_test, t_test) = get_data()
network = get_network()
print(network.accuracy(x_test[[6]], t_test[[6]]))

# (x_train, t_train), (x_test, t_test) = get_data()


# do()

