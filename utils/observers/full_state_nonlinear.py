from scipy.integrate import solve_ivp
from copy import deepcopy
import numpy as np

class Estimator:
    def __init__(self, ode, dt):
        self.ode = ode
        self.dt = dt

    def estimate(self, x_a, us):
        k = len(us)
        x = x_a
        xs = []
        for i in range(k):
            ts = (i * self.dt, (i + 1) * self.dt)
            u = us[i]
            res = solve_ivp(self.ode, ts, x, args=(u,))
            x = res.y[:, -1]
            xs.append(deepcopy(x))
        return np.array(xs)

