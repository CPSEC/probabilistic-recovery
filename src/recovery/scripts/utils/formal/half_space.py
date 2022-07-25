import matplotlib.pyplot as plt
import numpy as np


class HalfSpace:
    """
    normal vector - l
    The half space is defined as l^T x > b
    """
    def __init__(self, l: np.ndarray, b: [int, float]):
        self.l = l
        self.b = b
        self.dim = l.shape[0]

    def plot(self, x1, x2, fig=None):
        if self.dim != 2:
            return NotImplemented
        y1 = (self.b - x1 * self.l[0]) / self.l[1]
        y2 = (self.b - x2 * self.l[0]) / self.l[1]
        if fig is None:
            fig = plt.figure()
        X = [x1, x2]
        Y = [y1, y2]
        plt.plot(X, Y)
        if fig is None:
            plt.show()
        return fig

