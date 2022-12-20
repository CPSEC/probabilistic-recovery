import numpy as np
# for plot
from itertools import product
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from utils.formal.half_space import HalfSpace
from utils.formal.strip import Strip
from utils.formal.hyperplane import Hyperplane


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
        self.dim = c.shape[0]  # dimensions
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
        raise NotImplemented

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
        """
        Return:
            vertex - farthest vertex coordinate
            alpha  - direction (1/-1) for each generators
            gs_l   - generators to vertex
        """
        assert l.shape == (self.dim,)
        alpha = np.empty((len(self),))
        for i in range(len(self)):
            alpha[i] = -1 if self[i] @ l < 0 else 1
        gs_l = self.g @ np.diag(alpha)
        vertex = self.c + np.sum(gs_l, axis=1)
        return vertex, alpha, gs_l

    # algorithm 1
    def point_closest_to_hyperplane(self, hp: Hyperplane):
        z_star = self.c
        l = hp.l
        arrive = False
        g_num = len(self)
        alpha = np.zeros((g_num,))
        for i in range(g_num):
            alpha[i] = -1 if self[i] @ l < 0 else 1
            z_star_next = z_star+alpha[i]*self[i]
            if z_star_next @ l >= hp.b:
                alpha[i] = (hp.b-z_star@l)/(self[i]@l)
                z_star = z_star+alpha[i]*self[i]
                arrive = True
                break
            else:
                z_star = z_star_next
        return z_star, alpha, arrive

    # check if intersect with a half space
    def is_intersected(self, hs):
        if isinstance(hs, HalfSpace):
            return self.support(hs.l) >= hs.b
        if isinstance(hs, Strip):
            s1 = self.support(hs.l)
            s2 = self.support(-hs.l)
            return s1 >= hs.a and s2 <= hs.b
        if isinstance(hs, Hyperplane):
            s1 = self.support(hs.l)
            s2 = self.support(-hs.l)
            return s2 <= hs.b <= s1
        raise NotImplemented

    # get all vertices in order
    def to_V(self):
        # get all vertices
        n = self.g.shape[1]  # number of generators
        alpha = np.array(list(product(*list(zip([-1] * n, [1] * n))))).T
        v = self.c.reshape((-1, 1)) + self.g @ alpha  # all possible vertices for each column
        v = v.T  # one possible vertex per row
        v = v[ConvexHull(v).vertices, :]  # one vertex per row
        return v

    def plot(self, fig=None, color='orange'):
        v = self.to_V()
        if v.shape[1] != 2:  # only 2-d
            raise NotImplemented
        v = np.vstack((v, v[0]))  # close the polygon
        if fig is None:
            fig = plt.figure()
            plt.plot(v[:, 0], v[:, 1], color)
            plt.show()
        else:
            plt.plot(v[:, 0], v[:, 1], color)
        return fig

    # display routine by generators
    def show_routine(self, gs_l, fig=None):
        if self.dim != 2:
            raise NotImplemented
        routine = np.empty((len(self) + 1, self.dim), dtype=float)
        routine[0] = self.c
        for i in range(len(self)):
            routine[i + 1] = routine[i] + gs_l[:, i]
        if fig is None:
            fig = plt.figure()
        self.plot(fig)
        X = routine[:, 0]
        Y = routine[:, 1]
        for i in range(len(X) - 1):
            plt.arrow(X[i], Y[i], X[i + 1] - X[i], Y[i + 1] - Y[i], head_width=1.5, width=0.1,
                      length_includes_head=True, ec='g')
        if fig is None:
            plt.show()
        return fig

    # display routine by control inputs
    def show_control_effect(self, alpha, u_dim: int, head_width, line_width, fig=None):
        if self.dim != 2:
            raise NotImplemented
        # print(len(self), u_dim)
        gs_l = self.g @ np.diag(alpha)
        routine_num = len(self) // u_dim
        routine = np.empty((routine_num + 1, self.dim), dtype=float)
        routine[0] = self.c
        for i in range(routine_num):
            # print(gs_l[:, i*u_dim:(i+1)*u_dim], np.sum(gs_l[:, i*u_dim:(i+1)*u_dim], axis=0))
            u_effect = np.sum(gs_l[:, i * u_dim:(i + 1) * u_dim], axis=1)
            routine[i + 1] = routine[i] + u_effect
        if fig is None:
            fig = plt.figure()
        self.plot(fig)
        X = routine[:, 0]
        Y = routine[:, 1]
        if head_width is None:
            head_width = 1.5
            line_width = 0.1
        for i in range(len(X) - 1):
            plt.arrow(X[i], Y[i], X[i + 1] - X[i], Y[i + 1] - Y[i], head_width=head_width, width=line_width,
                      length_includes_head=True, ec='g')
        if fig is None:
            plt.show()
        return fig

    # call from one-step control input zonotope
    # for u, each row is a control input
    def alpha_to_control(self, alpha):
        u_dim = len(self)
        u_num = alpha.shape[0]//u_dim
        u = np.empty((u_num, u_dim), dtype=float)
        for i in range(u_num):
            u[i] = self.g @ alpha[i*u_dim:(i+1)*u_dim] + self.c
        return u

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
    print('support function along l:', z5.support(l))  # 2
    print('generators to farthest  vertex along l', z5.vertex_with_max_support(l))
    l = np.array([-1, 2])
    print('l=', l)
    print('support function along l:', z5.support(l))  # 5
    print('generators to farthest  vertex along l', z5.vertex_with_max_support(l))

    # check intersection
    l = np.array([-1, -1])
    bs = [4, 2, -2, -10]
    hss = [HalfSpace(l, b) for b in bs]
    res = [z5.is_intersected(hs) for hs in hss]
    print('check intersection', res)

    # plot zonotope
    v = z5.plot()

    # test alpha to control
    alpha = np.array([1, 1, -1, 1, -1, -1])
    u_lo = np.array([-1, -4])
    u_up = np.array([3, 4])
    U = Zonotope.from_box(u_lo, u_up)
    print(U)
    print(U.alpha_to_control(alpha))
