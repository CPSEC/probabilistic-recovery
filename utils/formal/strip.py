import numpy as np
from utils.formal.hyperplane import Hyperplane
import matplotlib.pyplot as plt

class Strip:
    """
    normal vector - l
    The half space is defined as a < l^T x < b
    """
    def __init__(self, l: np.ndarray, a: [int, float], b: [int, float]):
        self.l = l
        assert a < b
        self.a = a
        self.b = b
        self.dim = l.shape[0]

    def inverse_normal_vector(self):
        self.l = -self.l
        self.a, self.b = -self.b, -self.a

    # make sure l^T @ point < hp.b
    def point_to_strip(self, point):
        if self.l @ point > (self.b+self.a)/2:
            self.inverse_normal_vector()
            return True
        return False

    def center(self) -> Hyperplane:
        return Hyperplane(self.l, (self.a+self.b)/2)

    def in_strip(self, x):
        return self.a < self.l @ x < self.b

    def plot(self, x1, x2, fig=None):
        if self.dim != 2:
            return NotImplemented
        y1 = (self.b - x1 * self.l[0]) / self.l[1]
        y2 = (self.b - x2 * self.l[0]) / self.l[1]
        y3 = (self.a - x1 * self.l[0]) / self.l[1]
        y4 = (self.a - x2 * self.l[0]) / self.l[1]
        if fig is None:
            fig = plt.figure()
        X = [x1, x2]
        Y1 = [y1, y2]
        Y2 = [y3, y4]
        plt.plot(X, Y1)
        plt.plot(X, Y2)
        if fig is None:
            plt.show()
        return fig
