import numpy as np

class StateRecord:
    def __init__(self) -> None:
        self.xs = []
        self.us = []
    
    def record(self, x, u, t):
        assert t == len(self.xs)
        self.xs.append(x)
        self.us.append(u)

    def record_x(self, x, t):
        assert t == len(self.xs)
        self.xs.append(x)

    def record_u(self, u, t):
        assert t == len(self.us)
        self.us.append(u)

    def get_x(self, i):
        assert i < len(self.xs)
        return self.xs[i]

    def get_xs(self, start, end):
        return self.xs[start:end]

    def get_us(self, start, end):
        return self.us[start:end]

    def get_ys(self, C, start, end):
        xs = np.vstack(self.xs[start:end])
        ys = (C @ xs.T).T
        return ys
    