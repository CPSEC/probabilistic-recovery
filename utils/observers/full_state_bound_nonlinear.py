import numpy as np
import math
from copy import deepcopy
from .interval_helpers import *

class NonlinearEstimator:
    def __init__(self, ode, dt):
        self.ode = ode
        self.dt = dt

    def estimate(self, x_a, us, xs, unsafe_states_onehot):
        x_a, us = x_a.tolist(), us.tolist()
        if xs is not None:
            xs = xs.tolist()            
        x_a_interval = convert_1D_list_to_intervals(x_a)
        # print(f'inside estimate.... \n')
        # print(f'{x_a_interval=}, {us=}')
        x_0_interval = reach(self.ode, x_a_interval, us, xs, unsafe_states_onehot, step_size=self.dt)
        # print(f'{x_0_interval=}, {us=}')
        x_0_lo = [Int[0][0] for Int in x_0_interval]
        x_0_up = [Int[0][1] for Int in x_0_interval]
        x_0 = [midpoint(Int) for Int in x_0_interval]
        return np.array(x_0_lo), np.array(x_0_up), np.array(x_0)

    def get_deadline(self, x_a, safe_set_lo, safe_set_up, control: np.array, max_k):
        k = max_k
        breaked = False
        for i in range(max_k):
            i += 1
            control_series = np.array([control] * i)
            state_series = None # Not using safe states for deadline
            unsafe_states_onehot = [1] * len(x_a)
            x_0_lo, x_0_up, x_0 = self.estimate(x_a, control_series, state_series, unsafe_states_onehot)
            print(f'within deadline computation... @ {i=}/{max_k}, {x_0_lo=} and {x_0_up=} and {x_0=}\n')
            for j in range(np.size(x_a, 0)):
                if safe_set_lo[j] < x_0_lo[j] < safe_set_up[j] and safe_set_lo[j] < x_0_up[j] < safe_set_up[j]:
                    pass
                else:
                    breaked = True
                    break
            if breaked:
                k = i - 1
                break
        return k
