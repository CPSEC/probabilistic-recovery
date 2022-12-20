import numpy as np
import math
from copy import deepcopy


class Estimator:
    def __init__(self, Ad, Bd, max_k, epsilon=None, c=None):
        self.Ad = Ad
        self.Bd = Bd
        self.c = c
        self.n = self.Ad.shape[0]
        self.epsilon = epsilon
        self.max_k = max_k
        self.Ad_k = [np.eye(self.n)]
        for i in range(max_k):
            self.Ad_k.append(self.Ad_k[-1].dot(self.Ad))
        self.Ad_k_Bd = [i.dot(self.Bd) for i in self.Ad_k]
        if self.c is not None:
            self.Ad_k_c = [i.dot(self.c) for i in self.Ad_k]
        self.Ad_k_Ad_k_T = [i.dot(i.T) for i in self.Ad_k]
        self.epsilon_coef = []
        sqrt_term = np.zeros((self.n,))
        for k in range(max_k):
            for i in range(self.n):
                sqrt_term[i] += math.sqrt(self.Ad_k_Ad_k_T[k][i, i])
            self.epsilon_coef.append(deepcopy(sqrt_term))

    def estimate(self, x_a, us):
        # us: shape (length, m)
        k = len(us)
        assert k < self.max_k
        assert x_a.shape[0] == self.n
        assert self.epsilon is not None
        control_sum_term = np.zeros((self.n,))
        for j in range(k):
            if self.c is not None:
                control_sum_term += self.Ad_k_Bd[j] @ us[k - 1 - j] + self.Ad_k_c[j]
            else:
                control_sum_term += self.Ad_k_Bd[j] @ us[k - 1 - j]
        x_0 = self.Ad_k[k] @ x_a + control_sum_term
        e = np.ones((self.n,)) * self.epsilon * self.epsilon_coef[k]
        x_0_lo = x_0 - e
        x_0_up = x_0 + e
        return x_0_lo, x_0_up, x_0

    def estimate_wo_bound(self, x_a, us):
        # us: shape (length, m)
        k = len(us)
        assert k < self.max_k
        assert x_a.shape[0] == self.n
        control_sum_term = np.zeros((self.n,))
        for j in range(k):
            if self.c is not None:
                control_sum_term += self.Ad_k_Bd[j] @ us[k - 1 - j] + self.Ad_k_c[j]
            else:
                control_sum_term += self.Ad_k_Bd[j] @ us[k - 1 - j]
        x_0 = self.Ad_k[k] @ x_a + control_sum_term
        return x_0

    def get_deadline(self, x_a, safe_set_lo, safe_set_up, control: np.array, max_k):
        k = max_k
        breaked = False
        # print('control', control)
        for i in range(max_k):
            i += 1
            control_series = [control] * i
            # for j in range(i - 1):
            #     control_series = np.hstack((control_series, control))
            # # print("control_ser=", control_series)
            x_0_lo, x_0_up, x_0 = self.estimate(x_a, control_series)
            for j in range(np.size(x_a, 0)):
                if safe_set_lo[j] < x_0_lo[j] < safe_set_up[j] and safe_set_lo[j] < x_0_up[j] < safe_set_up[j]:
                    pass
                else:
                    breaked = True
                    break
            if breaked:
                k = i - 1
                # print('x_0_up', x_0_up)
                break
        # print('x_0_up', x_0_up)
        return k


if __name__ == '__main__':
    from scipy.signal import StateSpace, lsim

    A = np.array([[-10, 1], [-0.02, -2]])
    B = np.array([[0], [2]])
    C = np.array([1, 0])
    sys = StateSpace(A, B, C, 0)
    sysd = sys.to_discrete(0.02)

    est = Estimator(sysd.A, sysd.B, 20, 1e-7)
    x_0 = np.array([0, 0])
    control_lst = np.array([[1]] * 10)
    x0_lo, x0_up, x0 = est.estimate(x_0, control_lst)
    print(x0_lo, x0_up)

    t = np.arange(0, 0.22, 0.02)
    t, y, x = lsim(sys, np.array([[1]] * 11), t)
    print(t, x)
