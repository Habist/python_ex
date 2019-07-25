from matplotlib import pyplot as plt
import numpy as np


def coor_axis_graph(x_range_list):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)  # figure로 만든 공간안에 add_subplot을 통해 1칸을 만들고 그것을 객체화
    # Move left and lower axes to (0,0) point.
    ax.spines['left'].set_position('zero')  # spines는 matplotlib에 있다.
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes. 우축과 상축 색 없애서 안보이게
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Show ticks in the left and lower axes only. 좌축과 하축에만 눈금표시
    ax.xaxis.set_tick_params(bottom=True, top=False)
    ax.yaxis.set_tick_params(left=True, right=False)

    list_x = [];
    list_y = [];
    list = [];
    for (y,x) in x_range_list:
        # list_x.append(x)
        # list_y.append(y(x))
        plt.plot(x , y(x))
        # list.append([x , y(x)])

    # plt.plot(list)
    plt.show()

coor_axis_graph([(lambda x: x + 1 , np.arange(-3, 4, 1)), (lambda x: x - 1, np.arange(-5, 6, 1))])