from curve import HilbertCurve, ZCurve
import numpy as np
import matplotlib.pyplot as plt

'''
초기 SFC 생성 함수 : 이후 class 형태로 바꿀거임
'''
def build_init_coords(order, dimension, init_curve):
    if init_curve == 'zig-zag':
        whole_index = np.arange(2 ** (order * dimension))
        side = np.sqrt(2 ** (order * dimension)).astype(int)
        coords = np.array(list(map(lambda x: list([x // (side), x % (side)]), whole_index)))
    elif init_curve == 'hilbert':
        coords = HilbertCurve(dimension=dimension).getCoords(order=order)
    elif init_curve == 'z':
        coords = ZCurve(dimension=dimension).getCoords(order=order)
    return coords


'''
Grid (회색 선) 을 그릴 좌표를 써주는 함수
Arg : pmax 값
'''


def getGridCooridnates(num):
    grid_ticks = np.array([0, 2 ** num])
    for _ in range(num):
        temp = np.array([])
        for i, k in zip(grid_ticks[0::1], grid_ticks[1::1]):
            if i == 0:
                temp = np.append(temp, i)
            temp = np.append(temp, (i + k) / 2)
            temp = np.append(temp, k)
        grid_ticks = temp
    grid_ticks -= 0.5
    return grid_ticks


def showPoints(data, order, dim,  ax=None, index=True):
    ax = ax or plt.gca()
    pmax = np.ceil(np.log2(np.max(data)))
    pmax = pmax.astype(int)
    offset = 0.5
    cmin = 0
    cmax = 2 ** (pmax) - 1
    side = np.sqrt(2 ** (order * dim)).astype(int)

    grid_ticks = getGridCooridnates(pmax)

    ax.set_yticks(grid_ticks, minor=False)
    ax.set_xticks(grid_ticks, minor=False)
    plt.xlim(cmin - offset, cmax + offset)
    plt.ylim(cmin - offset, cmax + offset)
    ax.grid(alpha=0.5)

    if index:
        coordinates = np.array(list(map(lambda x: list([x // (side), x % (side)]), data)))
    else:
        coordinates = data

    ax.plot(coordinates[:, 0], coordinates[:, 1], 'o')
    print(f'pmax: {pmax}')


def showlineByIndexorder(data, index_to_coordinate, ax=None, index=True):
    ax = ax or plt.gca()

    if index:
        coordinates = index_to_coordinate[data]
    else:
        coordinates = data

    ax.plot(coordinates[:, 0], coordinates[:, 1], linewidth=1, linestyle='--')