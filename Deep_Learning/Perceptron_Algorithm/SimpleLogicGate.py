import numpy as np


#AND 회로 (인수 2개)
def AND_1(x1, x2):
    # x : 입력 신호
    # y : 출력 신호
    # w : 가중치
    # theta : 임계값
    # 가중치, 임계값은 현재 하이퍼 파라미터
    # 설명 : 입력 값에 가중치를 더한 값들이 임계값을 넘으면 신호 활성화
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# print(AND_1(0,0)) # 0을 출력
# print(AND_1(0,1)) # 0을 출력
# print(AND_1(1,0)) # 0을 출력
# print(AND_1(1,1)) # 1을 출력

def AND_2(x1, x2):
    # b : 편향 -> AND_1의 theta가 왼쪽으로 이항하여 -theta로 변하고 이를 편향 b로 설정
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# print(AND_2(0,0)) # 0을 출력
# print(AND_2(0,1)) # 0을 출력
# print(AND_2(1,0)) # 0을 출력
# print(AND_2(1,1)) # 1을 출력


#NAND 회로(인수 2개)
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # AND와 편향과 가중치를 다르게 함
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#OR 회로(인수 2개)
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#XOR 회로(인수 2개) // 단층 퍼셉트론으로는 구현 불가능해
#                   // 두개의 단층 퍼셉트론으로 구현
def XOR(x1, x2):
    s1 = NAND(x1, x2) # 단층 퍼셉트론
    s2 = OR(x1, x2) # 단층 퍼셉트론
    y = AND_2(s1, s2)
    return y

# print(XOR(0,0)) # 0을 출력
# print(XOR(0,1)) # 1을 출력
# print(XOR(1,0)) # 1을 출력
# print(XOR(1,1)) # 0을 출력













