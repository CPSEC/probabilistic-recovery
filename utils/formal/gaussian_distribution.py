import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


class GaussianDistribution:
    def __init__(self, miu: np.ndarray, sigma: np.ndarray):
        self.miu = miu
        self.sigma = sigma
        self.dim = miu.shape[0]

    # mask method in numpy
    __array_ufunc__ = None

    # '@' overload, linear transformation
    def __rmatmul__(self, other: np.ndarray):
        miu = other @ self.miu
        sigma = other @ self.sigma @ other.T
        return GaussianDistribution(miu, sigma)

    # '+' overload,
    # 1) add a bias
    # 2) add two independent gaussian distribution with same dim
    def __add__(self, other):
        if isinstance(other, np.ndarray):
            assert self.miu.shape == other.shape
            miu = self.miu + other
            return GaussianDistribution(miu, self.sigma)
        if isinstance(other, GaussianDistribution):
            assert self.miu.shape == other.miu.shape
            miu = self.miu + other.miu
            sigma = self.sigma + other.sigma
            return GaussianDistribution(miu, sigma)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        string = 'Gaussian Distribution:\n  miu:' + str(self.miu)
        string += '\n  sigma:' + str(self.sigma)
        return string

    def transformation_from_standard(self):
        return sqrtm(self.sigma)

    # generate points from this distribution
    # each column is a point
    def random(self, size):
        return sqrtm(self.sigma) @ np.random.randn(self.dim, size) + self.miu.reshape((-1, 1))

    def plot(self, x1, x2, y1, y2, fig=None):
        if self.dim != 2:
            return NotImplemented
        x, y = np.mgrid[x1:x2:.1, y1:y2:.1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = multivariate_normal(self.miu, self.sigma)
        z = rv.pdf(pos)
        if fig is None:
            fig = plt.figure()
        plt.contourf(x, y, z, levels=10, alpha=1)
        if fig is None:
            plt.show()
        return fig


if __name__ == "__main__":
    miu = np.array([1, 2])
    sigma = np.diag([2, 3])
    g1 = GaussianDistribution(miu, sigma)
    print('g1 =', g1)

    # linear transformation
    A = np.array([[1, 2], [3, 4]])
    b = np.array([-1, -1])
    g2 = A @ g1 + b
    print('A@g1+b =', g2)

    # sum of two independent ones
    g3 = g1 + g2
    print('g1+g2 =', g3)

    points = g3.random(4)
    print(points)
