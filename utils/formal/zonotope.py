import numpy as np
# for plot
from itertools import product
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from half_space import HalfSpace

class Zonotope:
    """
    Zonotope.
    parameters:
        c: center
        g: [g_0, g_1, ...]  generators
    """
    def __init__(self, c: np.ndarray, g: np.ndarray):
        self.c = c
        self.g = g
        self.dim = c.shape[0]   # dimensions
        assert self.g.shape[0] == self.dim

    def __str__(self):
        string = 'Zonotope\n  center:' + str(self.c)
        string += '\n  generators:'
        for i in range(len(self)):
            string += str(self[i])
        return string

    # '+' overload, Minkowski sum of two zonotopes
    def __add__(self, other):
        if isinstance(other, Zonotope):
            assert self.dim == other.dim
            c = self.c + other.c
            g = np.block([self.g, other.g])
            return Zonotope(c, g)
        if isinstance(other, np.ndarray):
            assert self.dim == other.shape[0]
            c = self.c + other
            return Zonotope(c, self.g)
        return NotImplemented

    # np.array + zonotope
    def __radd__(self, other):
        return self.__add__(other)

    # '+=' overload
    def __iadd__(self, other):
        assert self.dim == other.dim
        self.c += other.c
        self.g = np.block([self.g, other.g])
        return self

    # len() overload, get the number of generators
    def __len__(self):
        return self.g.shape[1]

    # [] slicing overload, get the key-th generator
    def __getitem__(self, item):
        assert 0 <= item < self.g.shape[1]
        return self.g[:, item]

    # mask method in numpy
    __array_ufunc__ = None

    # '@' overload, linear transformation
    def __rmatmul__(self, other: np.ndarray):
        assert other.shape[1] == self.dim
        c = other @ self.c
        g = other @ self.g
        return Zonotope(c, g)

    # support function along l direction
    def support(self, l: np.ndarray):
        assert l.shape == (self.dim,)
        A_T_l = self.g.T @ l
        rho_A_T_l = np.linalg.norm(A_T_l, ord=1)
        rho_c = l @ self.c
        return rho_c + rho_A_T_l

    # all generators contributing to the farther vertex along l
    def vertex_with_max_support(self, l: np.ndarray):
        assert l.shape == (self.dim,)
        u = self.g.copy()
        for i in range(len(self)):
            if self[i]@l < 0:
                u[:, i] *= -1
        vertex = self.c + np.sum(u, axis=1)
        return vertex, u

    # check if intersect with a half space
    def is_intersected(self, hs):
        if not isinstance(hs, HalfSpace):
            return NotImplemented
        return self.support(hs.l) >= hs.b

    # get all vertices in order
    def to_V(self):
        # get all vertices
        n = self.g.shape[1]    # number of generators
        alpha = np.array(list(product(*list(zip([-1]*n, [1]*n))))).T
        v = self.c.reshape((-1, 1)) + self.g @ alpha  # all possible vertices for each column
        v = v.T  # one possible vertex per row
        v = v[ConvexHull(v).vertices, :]  # one vertex per row
        return v

    def plot(self, fig=None):
        v = self.to_V()
        if v.shape[1] != 2:  # only 2-d
            return NotImplemented
        v = np.vstack((v, v[0]))  # close the polygon
        if fig is None:
            fig = plt.figure()
            plt.plot(v[:, 0], v[:, 1])
            plt.show()
        else:
            plt.plot(v[:, 0], v[:, 1])

    @classmethod
    def from_box(cls, lo: np.ndarray, up: np.ndarray):
        """
        initiate a zonotope from box
        lo: lower bound of the box
        up: upper bound of the box
        """
        c = (lo + up) / 2
        g_range = (up - lo) / 2
        g = np.diag(g_range)
        return cls(c, g)


if __name__ == '__main__':
    c1 = np.array([1, 2, 3], dtype=float)
    g1 = np.empty((3, 2))
    g1[:, 0] = np.array([1, 2, 3])
    g1[:, 1] = np.array([4, 5, 6])
    z1 = Zonotope(c1, g1)
    print("z1 =", z1)
    c2 = np.array([3, 1, 1], dtype=float)
    g2 = np.empty((3, 3))
    g2[:, 0] = np.array([9, 8, 7])
    g2[:, 1] = np.array([6, 5, 4])
    g2[:, 2] = np.array([3, 2, 1])
    z2 = Zonotope(c2, g2)
    print("z2 =", z2)

    # Minkowski sum
    z3 = z1 + z2
    print('z1+z2 = ', z3)

    # linear transformation
    A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    z4 = A @ z2
    print('A @ z2 =', z4)

    # support function
    c5 = np.array([1, 1], dtype=float)
    g5 = np.empty((2, 3))
    g5[:, 0] = np.array([1, 0])
    g5[:, 1] = np.array([0, -1])
    g5[:, 2] = np.array([1, 1])
    z5 = Zonotope(c5, g5)
    print('z5 =', z5)
    l = np.array([-1, -1])
    print('l=', l)
    print('support function along l:', z5.support(l))
    print('generators to farthest  vertex along l', z5.vertex_with_max_support(l))
    l = np.array([-1, 2])
    print('l=', l)
    print('support function along l:', z5.support(l))
    print('generators to farthest  vertex along l', z5.vertex_with_max_support(l))

    # check intersection
    l = np.array([-1, -1])
    bs = [4, 2, -2, -10]
    hss = [HalfSpace(l, b) for b in bs]
    res = [z5.is_intersected(hs) for hs in hss]
    print('check intersection', res)

    # plot zonotope
    v = z5.plot()







