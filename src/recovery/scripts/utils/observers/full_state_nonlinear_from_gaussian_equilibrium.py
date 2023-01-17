from copy import deepcopy
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from utils.control.linearizer import Linearizer
from utils.formal.gaussian_distribution import GaussianDistribution

class Estimator:
    def __init__(self, ode, n, m, dt, W:GaussianDistribution):
        self.ode = ode
        self.dt = dt
        self.n = n
        self.m = m
        self.W = W
        self.linearize = Linearizer(self.ode, self.n, self.m, self.dt)

    def estimate(self, x_0: GaussianDistribution, us):
        k = len(us)
        x = x_0.miu
        sysd = None  # recent linearized system
        As = []  #  0, k-1
        #  prod_Ai = A_{K-1} @ A_{K-2} @ ... @ A_0
        prod_Ai = np.eye(self.n)
        for i in range(k):
            # linearize at x_i, u_i to compute A_i, B_i, c_i
            ode_fixed_state = partial(self.ode, t=0, x=x)
            u_ss = fsolve(ode_fixed_state, us[i])
            # print(f'{u_ss=}')
            _, sysd = self.linearize.at(x, u_ss)
            As.append(deepcopy(sysd.A))

            # compute x_{i+1}=f(x_i, u_i)
            ts = (i * self.dt, (i + 1) * self.dt)
            res = solve_ivp(self.ode, ts, x, args=(u,))
            x = res.y[:, -1]

            # intermediate computations
            prod_Ai = As[i] @ prod_Ai

        #  sum_W = W + A_{k-1} W + A_{k-1} @ A_{k-2} W + ... + A_{k-1} @ A_{k-2} @ ... @ A_1 W
        sum_W = self.W
        pre_prod_Ai = np.eye(self.n)
        for i in range(k-1, 0, -1):
            pre_prod_Ai = pre_prod_Ai @ As[i]
            sum_W = sum_W + pre_prod_Ai @ self.W

        # compute the covariance of x_k
        sigma = prod_Ai @ x_0.sigma @ prod_Ai.T + sum_W.sigma
        x_k = GaussianDistribution(x, sigma)

        return x_k, sysd

