import numpy as np


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
        assert self.dim == other.dim
        c = self.c + other.c
        g = np.block([self.g, other.g])
        return Zonotope(c, g)

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
        return u


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








