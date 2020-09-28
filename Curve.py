import numpy as np


class HilbertCurve():
    def __init__(self, dimension):
        self.DIM = dimension

    # convert (x,y) to d
    def xy2d(self, n, x, y):
        d = 0
        s = n // 2
        while s > 0:
            rx = ((x & s) > 0);
            ry = ((y & s) > 0);
            d += s * s * ((3 * rx) ^ ry)
            x, y = self.rot(n, x, y, rx, ry)
            s = s // 2
        return d

    def d2xy(self, n, d):
        t = d
        x = 0
        y = 0
        s = 1
        while s < n:
            rx = 1 & t // 2
            ry = 1 & t ^ rx
            x, y = self.rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t = t // 4
            s *= 2
        return [x, y]

    def rot(self, n, x, y, rx, ry):
        if (ry == 0):
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            t = x
            x = y
            y = t
        return x, y

    def getCoords(self, order):
        N = 2 ** (order * self.DIM)
        coordinates = list(map(self.d2xy, [N] * (N), range(N)))
        return coordinates


'''
생성된 SFC와 비교하기 위한 Z curve
'''


class ZCurve():
    def __init__(self, dimension):
        self.DIM = dimension

    def part1by1(self, n):
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    def unpart1by1(self, n):
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0f0f0f0f
        n = (n ^ (n >> 4)) & 0x00ff00ff
        n = (n ^ (n >> 8)) & 0x0000ffff
        return n

    def part1by2(self, n):
        n &= 0x000003ff
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n << 8)) & 0x0300f00f
        n = (n ^ (n << 4)) & 0x030c30c3
        n = (n ^ (n << 2)) & 0x09249249
        return n

    def unpart1by2(self, n):
        n &= 0x09249249
        n = (n ^ (n >> 2)) & 0x030c30c3
        n = (n ^ (n >> 4)) & 0x0300f00f
        n = (n ^ (n >> 8)) & 0xff0000ff
        n = (n ^ (n >> 16)) & 0x000003ff
        return n

    # 2 차원 데이터를 비트로 변환하고 교차 생성
    def interleave2(self, x, y):
        return self.part1by1(x) | (self.part1by1(y) << 1)

    # 교차 생성된 값을 2 차원 데이터로 되돌림
    def deinterleave2(self, n):
        return [self.unpart1by1(n), self.unpart1by1(n >> 1)]

    def interleave3(self, x, y, z):
        return self.part1by2(x) | (self.part1by2(y) << 1) | (self.part1by2(z) << 2)

    def deinterleave3(self, n):
        return [self.unpart1by2(n), self.unpart1by2(n >> 1), self.unpart1by2(n >> 2)]

    def getCoords(self, order):
        # temp_index = np.arange(2**(self.DIM * order))
        coords = list(map(self.deinterleave2, np.arange(2 ** (self.DIM * order))))
        return np.array(coords)
