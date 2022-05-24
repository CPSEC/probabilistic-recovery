import numpy as np

class Strip:
    """
    normal vector - l
    The half space is defined as a < l^T x < b
    """
    def __init__(self, l: np.ndarray, a:[int, float], b: [int, float]):
        self.l = l
        self.a = a
        self.b = b
        self.dim = l.shape[0]