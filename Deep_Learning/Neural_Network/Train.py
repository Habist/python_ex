import pickle
from Optimization import *


class Train:
    def __init__(self, get_data, network, iters_num = 5000,batch_size = 100,
                 learning_rate= 0.01, optimizer='Adam', optimizer_lr = 0.01, use_save = False):
        self.network = network
        self.network_name = network.__class__.__name__
        (self.x_train, self.t_train), (self.x_test, self.t_test) = get_data()
        self.iters_num = iters_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # optimizer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum,
                                'adagrad': AdaGrad, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](optimizer_lr)
        self.use_save = use_save

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)

    def train_step(self):
        for i in range(self.iters_num):
            batch_mask = np.random.choice(self.train_size, self.batch_size)  # 어레이 인덱스를 범위 내에서 무작위로 생성
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            # grads = network.numerical_gradient(x_batch, t_batch)     # 수치미분을 사용해 신경망의 손실함수에 대한 weight, bias 미분값을 구함
            grads = self.network.gradient(x_batch, t_batch)  # 오차역전파를 사용해 매개변수 구함
            self.optimizer.update(self.network.params, grads)  # 매개변수 갱신

            # for key in ('W1', 'b1', 'W2', 'b2'):
            #     network.params[key] -= learning_rate * grad[key]    # 미분값을 학습률과 곱하여 기존값에서 빼는 방식으로 갱신

            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)

            if i % self.iter_per_epoch == 0:
                train_acc = self.network.accuracy(self.x_train, self.t_train)
                test_acc = self.network.accuracy(self.x_test, self.t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    def save_network(self):
        with open("./saveNetwork/"+ self.network_name + ".pkl", "wb") as file:
            pickle.dump(self.network, file)
            file.close()

    def load_network(self, file_name):
        with open("./saveNetwork/" + file_name + ".pkl", "rb") as file:
            self.network = pickle.load(file)

    def predict(self,idx):
        y = self.network.predict(self.x_test[[idx]])
        y = np.argmax(y, axis=1)
        if y == self.t_test[[idx]]:
            print('정답')
        else:
            print('오답')
        print('예측 결과 : ' + y[0].__str__())
        print('실제 답 : ' + self.t_test[[idx]][0].__str__())

