from scipy.integrate import solve_ivp
from copy import deepcopy
import numpy as np

from utils.control.linearizer import Linearizer
from utils.formal.gaussian_distribution import GaussianDistribution
import time

class Estimator:
    def __init__(self, ode, n, m, dt, W:GaussianDistribution, euler=True, kf=None, jfx=None, jfu=None):
        self.ode = ode
        self.dt = dt
        self.n = n
        self.m = m
        self.W = W
        self.linearize = Linearizer(self.ode, self.n, self.m, self.dt, jfx, jfu)
        self.euler = euler
        self.kf = kf
        if kf is not None:
            self.P = np.zeros((n, n))
        

    def estimate(self, x_0: GaussianDistribution, us, ys=None):
        if self.kf is None:
            return self.estimate_ol(x_0, us)
        else:
            return self.estimate_cl(x_0, us, ys)
        
    def estimate_ol(self, x_0: GaussianDistribution, us):
        k = len(us)
        x = x_0.miu
        sysd = None  # recent linearized system
        As = []  #  0, k-1
        #  prod_Ai = A_{K-1} @ A_{K-2} @ ... @ A_0
        prod_Ai = np.eye(self.n)
        for i in range(k):
            # linearize at x_i, u_i to compute A_i, B_i, c_i
            u = us[i]
            sysd = self.linearize.at(x, u)
            As.append(deepcopy(sysd.A))

            # compute x_{i+1}=f(x_i, u_i)
            ts = (i * self.dt, (i + 1) * self.dt)
            t_init = time.time()
            if not self.euler:
                res = solve_ivp(self.ode, ts, x, args=(u,))
                x = res.y[:, -1]
            else:
                x = self.solve_euler(ts, x, u)
            t_solve = time.time() - t_init
            # print(t_solve)

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
    
    def estimate_cl(self, x_0: GaussianDistribution, us, ys): 
        k = len(us)
        x = x_0.miu
        sysd = None  # recent linearized system
        As = []  #  0, k-1
        #  prod_Ai = A_{K-1} @ A_{K-2} @ ... @ A_0
        prod_Ai = np.eye(self.n)
        for i in range(k):
            # linearize at x_i, u_i to compute A_i, B_i, c_i
            u = us[i]
            y = ys[i]
            sysd = self.linearize.at(x, u)
            As.append(deepcopy(sysd.A))
            # print(sysd.A)

            # compute x_{i+1}=f(x_i, u_i)
            x, self.P = self.kf.one_step(x, self.P, u, y)

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
    
    def solve_euler(self, t, x, u):
        div = 10
        dt = t[-1] - t[0]
        for _ in range(div):
            x = x + self.ode(0, x, u) * dt / div
        return x

