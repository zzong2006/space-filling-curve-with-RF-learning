import numpy as np
import matplotlib.pyplot as plt
from utils import *
import configparser


class CurveEnvironment:
    def __init__(self, order=3, dim=2, data_size=10, init_curve='zig-zag'):

        self.order = order
        self.dim = dim
        self.data_size = data_size
        self.total_grid = 2 ** (order * dim)
        self.side = int(np.sqrt(self.total_grid))  # grid 세로 또는 가로 개수
        self.init_curve = init_curve

        # 임의의 데이터 분포 생성
        scan_index = np.random.choice(self.total_grid, size=data_size, replace=False)
        self.random_data = np.array(
            list(map(lambda x: list([x // self.side, x % self.side]), scan_index)))  # 생성된 데이터의 좌표 구성

    def print_curve(self, ):
        fig, ax = plt.subplots(1, figsize=(10, 10))

        show_points(self.random_data, self.side, ax, index=False)

        if self.init_curve == 'hilbert':
            temp_curve = HilbertCurve(self.dim)
            show_line_by_index_order(np.array(temp_curve.getCoords(self.order)), ax)
        elif self.init_curve == 'zig-zag':
            grid_index = np.arange(self.total_grid)
            show_line_by_index_order(grid_index, ax)
        plt.show(block=True)


if '__main__' == __name__:
    test_env = CurveEnvironment()
    test_env.print_curve()
