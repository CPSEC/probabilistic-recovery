import matplotlib.pyplot as plt
import numpy as np
from zonotope import Zonotope
from gaussian_distribution import GaussianDistribution
from half_space import HalfSpace


class ReachableSet:
    """
    get the Reachable set
    A, B - from discrete-time system
    """
    def __init__(self, A, B, U: Zonotope, W: GaussianDistribution, max_step=50):
        self.max_step = max_step
        self.A_k = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_k.append(A @ self.A_k[-1])
        self.A_k_B_U = [val @ B @ U for val in self.A_k]
        self.A_k_W = [val @ W for val in self.A_k]
        self.bar_u_k = [U, self.A_k_B_U[0]]
        self.bar_w_k = [W, self.A_k_W[0]]
        for i in range(1, max_step):
            self.bar_u_k.append(self.A_k_B_U[i]+self.bar_u_k[-1])
            self.bar_w_k.append(self.A_k_W[i]+self.bar_w_k[-1])

    def init(self, x_0: GaussianDistribution, hs: HalfSpace):
        self.x_0 = x_0
        self.hs = hs

    def reachable_set_wo_noise(self, k: int) -> Zonotope:
        x_0 = self.x_0.miu
        X_k = self.A_k[k]@x_0 + self.bar_u_k[k]
        return X_k

    def first_intersection(self) -> ([int, None], Zonotope):
        for i in range(1, self.max_step):
            X_k = self.reachable_set_wo_noise(i)
            if X_k.is_intersected(self.hs):
                return i, X_k
        return None, X_k



if __name__ == '__main__':
    u_lo = np.array([-1, -4])
    u_up = np.array([3, 4])
    U = Zonotope.from_box(u_lo, u_up)
    print(U)
    # U.plot()

    A = np.array([[1, 1], [0, 2]])
    B = np.array([[2, 0], [0, 1]])
    W = GaussianDistribution(np.array([0, 0]), np.eye(2))
    reach = ReachableSet(A, B, U, W, max_step=5)
    x_0 = GaussianDistribution(np.array([5, 5]), np.eye(2))
    hs = HalfSpace(np.array([1, 1]), 100)
    reach.init(x_0, hs)
    X_1 = reach.reachable_set_wo_noise(1)
    print(X_1)
    # X_1.plot()

    X_2 = reach.reachable_set_wo_noise(2)
    print(X_2)
    # X_2.plot()

    X_3 = reach.reachable_set_wo_noise(3)
    # X_3.plot()

    k, X_k = reach.first_intersection()
    print('k =', k)
    vertex, alpha, gs_l = X_k.vertex_with_max_support(hs.l)
    fig = plt.figure()
    X_k.show_routine(gs_l, fig)
    hs.plot(30, 80, fig)
    plt.show()

    vertex, alpha, gs_l = X_2.vertex_with_max_support(hs.l)
    fig = plt.figure()
    # X_2.show_routine(gs_l, fig)
    X_2.plot(fig)
    hs.plot(30, 80, fig)
    plt.show()

    # # print(reach.A_k)
    # for val in reach.A_k_B_U:
    #     print(val)
    # print(reach.A_k_B_U)