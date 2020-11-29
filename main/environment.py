from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import configparser
import torch


class CurveEnvironment:
    """
        Curve 환경, 강화 학습 모델이 생성한 action을 기반으로 커브의 순서를 바꾸고, 바뀐 커브를 기반으로 reward를 측정
    """

    def __init__(self, order=3, dim=2, data_size=10, init_curve='zig-zag', normalize=True, life=10, seed=1234):
        """
        
        :param order: Curve iteration 개수
        :param dim: 차원 수
        :param data_size: 활성화 데이터 개수
        :param init_curve: 초기 커브, 이 커브의 순서를 바꾸면서 최적의 커브를 찾음
        :param normalize: 주어진 coordinate를 normalize 할것인지?
        :param life: 한 episode 당 주어지는 목숨 
        :param seed: 활성화 데이터 생성 시드
        """
        self.order = order
        self.dim = dim
        self.data_size = data_size
        self.total_grid = 2 ** (order * dim)
        self.side = int(np.sqrt(self.total_grid))  # grid 세로 또는 가로 개수
        self.init_curve = init_curve
        self.normalized = normalize
        self.debug = dict()         # 디버그용 정보가 담긴 dictionary. 주로, cost 정보를 담음


        np.random.seed(seed)
        # 임의의 데이터 분포 생성
        self.data_index = np.random.choice(self.total_grid, size=data_size, replace=False)
        self.data_coord = np.array(
            list(map(lambda x: list([x // self.side, x % self.side]), self.data_index)))  # 생성된 데이터의 좌표 구성

        # episode 종료 기준
        self.life = life  # life 가 0에 도달하면 episode 종료
        self.ori_life = life

        # 커브 생성
        self.curve_coord = self.reset()

        # reward 측정용 기준
        self.min_cost = self.get_l2_norm_locality()
        self.prev_cost = self.min_cost

    @staticmethod
    def normalize_state(state):
        min_val = np.min(state, axis=0, keepdims=True)
        max_val = np.max(state, axis=0, keepdims=True)
        state = (state - min_val) / (max_val - min_val)

        return state

    def build_init_coords(self):
        """
        초기 곡선 타입에 따른 n 차원 좌표 list를 만드는 함수, list 내 좌표 배치 순서는 곡선 타입을 따름
        :return:
        """
        coords = None
        try:
            if self.init_curve == 'zig-zag':
                whole_index = np.arange(self.total_grid)
                coords = np.array(list(map(lambda x: list([x // self.side, x % self.side]), whole_index)))
            elif self.init_curve == 'hilbert':
                coords = HilbertCurve(dimension=self.dim).getCoords(order=self.order)
            elif self.init_curve == 'z':
                coords = ZCurve(dimension=self.dim).getCoords(order=self.order)
            else:
                raise Exception('Curve type must be "zig-zag" or "hilbert" or "z".')
        except Exception as e:
            print(e)
        finally:
            return coords

    def reset(self):
        """
        n 차원 곡선 좌표 list를 생성하고, 해당 좌표의 활성화 데이터 여부를 표시하는 함수
        또한 reward 측정을 위한 기준을 초기화함
        :return: 
        """

        self.curve_coord = self.build_init_coords()  # 곡선을 n 차원 좌표 list로 구성
        avail = np.zeros(shape=(self.total_grid, 1), dtype=np.int)

        # 이미 생성된 활성화 데이터의 좌표가 일치되는 곳을 활성화
        for index in map(lambda x: np.where(np.all(self.curve_coord == x, axis=1)), self.data_coord):
            avail[index] = 1  # 활성화 데이터 여부 표시

        self.curve_coord = np.concatenate((avail, self.curve_coord), axis=1)

        if self.normalized:  # do feature scaling
            self.curve_coord = CurveEnvironment.normalize_state(self.curve_coord)

        self.min_cost = self.get_l2_norm_locality()
        self.prev_cost = self.min_cost
        self.life = self.ori_life

        return self.curve_coord

    def plot_curve(self, ):
        fig, ax = plt.subplots(1, figsize=(10, 10))

        show_points(self.data_coord, self.side, ax, index=False)

        if self.init_curve == 'hilbert':
            temp_curve = HilbertCurve(self.dim)
            show_line_by_index_order(np.array(temp_curve.getCoords(self.order)), ax)
        elif self.init_curve == 'zig-zag':
            grid_index = np.arange(self.total_grid)
            show_line_by_index_order(grid_index, ax)
        plt.show(block=True)

    def get_l2_norm_locality(self):
        """
        l2 norm ratio locality 측정 함수
        sum(1 - (l2 norm/ l1 norm)) 의 형태
        
        :return: 
        """
        avail_data = []
        for idx, point in enumerate(self.curve_coord):
            if point[0] == 1:  # 활성화 데이터인 경우
                avail_data.append([point[1], point[2], idx])
        cost = 0

        # 활성화된 데이터만 모음, 결과는 (x, y, 데이터 순서)
        for (x, y) in combinations(avail_data, 2):
            dist_2d = np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
            dist_1d = np.abs(x[2] - y[2])
            # Locality Ratio 가 1과 가까운지 측정
            cost += np.abs(1 - (dist_1d / dist_2d))

        return cost

    def get_reward(self):
        """
        보상 측정 함수, l2_norm_locality가 감소한 경우 positive reward를 부여한다. 그 외에는 0 또는 negative reward
        :return: 
        """
        curr_cost = self.get_l2_norm_locality()
        reward = 0
        self.debug['cost'] = curr_cost

        if self.min_cost < curr_cost:  # 최소 cost 보다 작아지지 못할 경우
            if self.prev_cost < curr_cost:
                self.life -= 1
                reward = -1
            elif self.prev_cost > curr_cost:  # 최소 cost 보다 작아지지 못했지만, 이전 커브 cost 보다는 작아졌을 경우
                reward = 0
            else:
                reward = 0
        elif self.prev_cost == curr_cost:
            reward = 0
        else:
            reward = max(1, abs(curr_cost - self.min_cost))
            self.min_cost = curr_cost  # 최소 cost 갱신
        self.prev_cost = curr_cost  # 이전 cost 갱신

        return reward

    def step(self, action: tuple):
        a, b = action
        self.curve_coord[[a, b]] = self.curve_coord[[b, a]]     # grid 순서 swap
        reward = self.get_reward()

        done = False
        if self.life == 0:
            done = True

        return self.curve_coord, reward, done, self.debug


if '__main__' == __name__:
    test_env = CurveEnvironment()
    for curve_name in ['z', 'hilbert', 'zig-zag']:
        test_env = CurveEnvironment(init_curve=curve_name)
        print(test_env.get_l2_norm_locality())
    print(test_env.get_reward())
