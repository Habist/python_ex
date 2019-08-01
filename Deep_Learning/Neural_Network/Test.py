import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist, init_network
from SimpleNeuralNetwork import predict

def image_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False, one_hot_label=False) #정규화 전처리 True
    return x_test, t_test


def do_each():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    # wrong_idx = []
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 확률이 가장 높은 인덱스 저장
        # print(p, t[i], i)
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

networ1k = init_network()

W1, W2, W3 = networ1k['W1'], networ1k['W2'], networ1k['W3']
b1, b2, b3 = networ1k['b1'], networ1k['b2'], networ1k['b3']

print(networ1k.keys())
print(networ1k)
