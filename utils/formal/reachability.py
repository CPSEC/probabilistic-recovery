import matplotlib.pyplot as plt
import numpy as np
from zonotope import Zonotope
from gaussian_distribution import GaussianDistribution
from utils.formal.half_space import HalfSpace
from utils.formal.strip import Strip


class ReachableSet:
    """
    get the Reachable set + Distribution
    A, B - from discrete-time system
    """

    def __init__(self, A, B, U: Zonotope, W: GaussianDistribution, max_step=50):
        self.max_step = max_step
        self.u_dim = len(U)
        self.A_k = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_k.append(A @ self.A_k[-1])
        self.A_k_B_U = [val @ B @ U for val in self.A_k]
        self.A_k_W = [val @ W for val in self.A_k]
        self.bar_u_k = [U, self.A_k_B_U[0]]
        self.bar_w_k = [W, self.A_k_W[0]]
        for i in range(1, max_step):
            self.bar_u_k.append(self.A_k_B_U[i] + self.bar_u_k[-1])
            self.bar_w_k.append(self.A_k_W[i] + self.bar_w_k[-1])

    def init(self, x_0: GaussianDistribution, s: Strip):
        self.x_0 = x_0
        self.s = s
        self.hp = s.center()

    def reachable_set_wo_noise(self, k: int) -> Zonotope:
        x_0 = self.x_0.miu
        X_k = self.A_k[k] @ x_0 + self.bar_u_k[k]
        if self.s.point_to_strip(X_k.c):
            self.hp = s.center()
        return X_k

    def first_intersection(self) -> ([int, None], Zonotope):
        for i in range(1, self.max_step):
            X_k = self.reachable_set_wo_noise(i)
            if X_k.is_intersected(self.hs):
                return i, X_k
        return None, X_k

    def distribution(self, vertex: np.ndarray, k: int):
        return vertex + self.bar_w_k[k]

    def reachable_set_k(self, k: int, fig_setting=None):
        X_k = self.reachable_set_wo_noise(k)
        z_star, alpha = X_k.point_closest_to_hyperplane(reach.hp)
        D_k = self.distribution(z_star, k)
        if not fig_setting is None:
            fig = plt.figure()
            X_k.plot(fig)
            s.plot(fig_setting['x1'], fig_setting['x2'], fig)
            X_k.show_control_effect(alpha, self.u_dim, fig)
            plt.show()
        return X_k, D_k, z_star, alpha


if __name__ == '__main__':
    u_lo = np.array([-1, -4])
    u_up = np.array([3, 4])
    U = Zonotope.from_box(u_lo, u_up)
    print(U)
    # U.plot()

    A = np.array([[1, 1], [0, 2]])
    B = np.array([[2, 0], [0, 1]])
    W = GaussianDistribution(np.array([0, 0]), 0.3 * np.eye(2))
    reach = ReachableSet(A, B, U, W, max_step=5)
    x_0 = GaussianDistribution(np.array([5, 5]), np.eye(2))

    s = Strip(np.array([1, 1]), a=100, b=120)
    reach.init(x_0, s)

    fig_setting = {'x1': 30, 'x2': 80}
    X_k, D_k, z_star, alpha = reach.reachable_set_k(1, fig_setting)

    X_k, D_k, z_star, alpha = reach.reachable_set_k(2, fig_setting)

    X_k, D_k, z_star, alpha = reach.reachable_set_k(3, fig_setting)
