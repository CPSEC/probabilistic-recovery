import matplotlib.pyplot as plt
import numpy as np
from utils.formal.zonotope import Zonotope
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.half_space import HalfSpace
from utils.formal.strip import Strip


class ReachableSet:
    """
    get the Reachable set + Distribution
    A, B - from discrete-time system
    """

    def __init__(self, A, B, U: Zonotope, W: GaussianDistribution, max_step=50, c=None):
        self.ready = False
        self.max_step = max_step
        self.u_dim = len(U)
        self.A = A
        self.B = B
        self.c = c
        self.A_k = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_k.append(A @ self.A_k[-1])
        self.A_k_B = [val @ B for val in self.A_k]
        self.A_k_B_U = [val @ B @ U for val in self.A_k]
        self.A_k_W = [val @ W for val in self.A_k]
        if self.c is not None:
            self.Ad_k_c = [i.dot(self.c) for i in self.A_k]
            self.bar_c_k = [self.c, self.Ad_k_c[0]]
        self.bar_u_k = [U, self.A_k_B_U[0]]
        self.bar_w_k = [W, self.A_k_W[0]]
        for i in range(1, max_step):
            self.bar_u_k.append(self.A_k_B_U[i] + self.bar_u_k[-1])
            self.bar_w_k.append(self.A_k_W[i] + self.bar_w_k[-1])
            if self.c is not None:
                self.bar_c_k.append(self.Ad_k_c[i] + self.bar_c_k[-1])

    def init(self, x_0: GaussianDistribution, s: Strip):
        self.x_0 = x_0
        self.s = s
        self.hp = s.center()
        self.ready = True

    def reachable_set_wo_noise(self, k: int) -> Zonotope:
        x_0 = self.x_0.miu
        if self.c is not None:
            X_k = self.A_k[k] @ x_0 + self.bar_u_k[k] + self.bar_c_k[k]
        else:
            X_k = self.A_k[k] @ x_0 + self.bar_u_k[k]
        if self.s.point_to_strip(X_k.c):
            self.hp = self.s.center()
        return X_k

    def state_reconstruction(self, us) -> GaussianDistribution:
        k = len(us)
        x = self.x_0.miu
        for u in us:
            if self.c is not None:
                x = self.A @ x + self.B @ u + self.c
            else:
                x = self.A @ x + self.B @ u
        return self.distribution(x, k)

    def first_intersection(self) -> ([int, None], Zonotope):
        for i in range(1, self.max_step):
            X_k = self.reachable_set_wo_noise(i)
            if X_k.is_intersected(self.hs):
                return i, X_k
        return None, X_k

    def distribution(self, vertex: [np.ndarray, GaussianDistribution], k: int):
        zstar_cov = self.A_k[k] @ self.x_0.sigma @ self.A_k[k].T
        zstar_distribution = GaussianDistribution(vertex, zstar_cov)
        return zstar_distribution + self.bar_w_k[k]  # count in x_0 covariance

    def reachable_set_k(self, k: int):
        X_k = self.reachable_set_wo_noise(k)
        z_star, alpha, arrive = X_k.point_closest_to_hyperplane(self.hp)
        D_k = self.distribution(z_star, k)
        P = D_k.prob_in_strip(self.s)
        return X_k, D_k, z_star, alpha, P, arrive

    def plot(self, X_k: Zonotope, D_k: GaussianDistribution, alpha, fig_setting):
        fig = plt.figure()
        if fig_setting['distribution'] and 'x1' in fig_setting and 'x2' in fig_setting and 'y1' in fig_setting and \
                'y2' in fig_setting:
            x1, x2, y1, y2 = fig_setting['x1'], fig_setting['x2'], fig_setting['y1'], fig_setting['y2']
            D_k.plot(x1, x2, y1, y2, fig)
        if fig_setting['zonotope']:
            X_k.plot(fig)
        if fig_setting['strip'] and 'x1' in fig_setting and 'x2' in fig_setting:
            self.s.plot(fig_setting['x1'], fig_setting['x2'], fig)
        if fig_setting['routine']:
            head_width = line_width = None
            if 'head_width' in fig_setting:
                head_width = fig_setting['head_width']
                line_width = fig_setting['width']
            X_k.show_control_effect(alpha, self.u_dim, head_width, line_width, fig)
        if 'x1' in fig_setting and 'x2' in fig_setting:
            plt.xlim((fig_setting['x1'], fig_setting['x2']))
        if 'y1' in fig_setting and 'y2' in fig_setting:
            plt.ylim((fig_setting['y1'], fig_setting['y2']))
        plt.show()

    def given_k(self, max_k: int):
        if not self.ready:
            print('Init before recovery!')
            raise RuntimeError
        dummy_res = (None, None, None, None, 0, False)
        reach_res = [dummy_res]
        arrived = False
        max_P = 0
        for i in range(1, max_k + 1):
            res = self.reachable_set_k(i)
            reach_res.append(res)
            # X_k, D_k, z_star, alpha, P, arrive = res
            if res[4] > max_P:
                max_P = res[4]
            if res[5] and not arrived:
                arrived = True
                break
        all_res = [val for val in reach_res if val[4] == max_P]
        res = all_res[-1]
        k = reach_res.index(res)
        return k, *res

    def given_P(self, P_given: float, max_k: int):
        if not self.ready:
            print('Init before recovery!')
            raise RuntimeError
        satisfy = False
        i = 0
        X_k = D_k = z_star = alpha = P = arrive = None
        for i in range(1, max_k + 1):
            X_k, D_k, z_star, alpha, P, arrive = self.reachable_set_k(i)
            if P > P_given:
                satisfy = True
                break
            if arrive == True:
                break
        return i, satisfy, X_k, D_k, z_star, alpha, P, arrive

    def maintain_once(self, P_given: float):
        if not self.ready:
            print('Init before recovery!')
            raise RuntimeError
        res = self.reachable_set_k(1)
        if res[4] >= P_given:
            return True, *res
        else:
            return False, *res


if __name__ == '__main__':
    u_lo = np.array([-1, -2])
    u_up = np.array([3, 4])
    U = Zonotope.from_box(u_lo, u_up)
    print(U)
    # U.plot()

    A = np.array([[1, 1], [0, 2]])
    B = np.array([[2, 0], [0, 1]])
    W = GaussianDistribution.from_standard(miu=np.array([0, 0]), C=np.diag([0.3, 0.3]))
    reach = ReachableSet(A, B, U, W, max_step=5)
    x_0 = GaussianDistribution(np.array([5, 5]), np.eye(2))

    s = Strip(np.array([1, 1]), a=100, b=120)
    reach.init(x_0, s)

    fig_setting = {'x1': 0, 'x2': 80, 'y1': 0, 'y2': 90,
                   'strip': True, 'routine': True,
                   'zonotope': True, 'distribution': True}
    X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(1)
    # reach.plot(X_k, D_k, alpha, fig_setting)
    print('i=', 1, 'P=', P, 'z_star=', z_star, 'arrive=', arrive, 'alpha=', alpha)

    X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(2)
    # reach.plot(X_k, D_k, alpha, fig_setting)
    print('i=', 2, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)

    X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(3)
    # reach.plot(X_k, D_k, alpha,fig_setting)
    print('i=', 3, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)

    i, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=0.9, max_k=10)
    print('i=', i, 'found=', satisfy)
    print('i=', i, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)
    print('alpha=', alpha)
    # reach.plot(X_k, D_k, alpha, fig_setting)
    rec_u = U.alpha_to_control(alpha)
    print('rec_u =', rec_u)

    x1 = A @ np.array([5, 5]) + B @ rec_u[0]
    x2 = A @ x1 + B @ rec_u[1]
    x3 = A @ x2 + B @ rec_u[1]
    print('x3 =', x3)

    # k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(5)
    # print('i=', i, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)
    # reach.plot(X_k, D_k, alpha, fig_setting)
