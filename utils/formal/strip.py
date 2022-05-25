import numpy as np
from utils.formal.hyperplane import Hyperplane


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

    # make sure l^T @ point < self.b
    def point_to_strip(self, point):
        if self.l @ point > self.b:
            self.inverse_normal_vector()

    def center(self) -> Hyperplane:
        return Hyperplane()
