import numpy as np


class Hyperplane:
    def __init__(self, l: np.ndarray, b: [int, float]):
        self.l = l
        self.b = b
        self.dim = l.shape[0]

    def inverse_normal_vector(self):
        self.l = -self.l
        self.b = -self.b
