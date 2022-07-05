# Ref: [1] J. M. Snider, “Automatic Steering Methods for Autonomous Automobile Path Tracking,” p. 78.

import numpy as np

from utils import Simulator
from utils.controllers.LQR import LQR

cf_ = 155494.663
cr_ = 155494.663
wheelbase_ = 2.852
steer_ratio_ = 16
steer_single_direction_max_degree_ = 470.0
mass_fl = 520
mass_fr = 520
mass_rl = 520
mass_rr = 520
mass_front = mass_fl + mass_fr
mass_rear = mass_rl + mass_rr
mass_ = mass_front + mass_rear
lf_ = wheelbase_ * (1.0 - mass_front / mass_)
lr_ = wheelbase_ * (1.0 - mass_rear / mass_)
iz_ = lf_ * lf_ * mass_front + lr_ * lr_ * mass_rear
vx = 5

A = np.zeros((4, 4), dtype=np.float32)
A[0, 1] = 1.0
A[1, 1] = -(cf_ + cr_) / mass_ / vx
A[1, 2] = (cf_ + cr_) / mass_
A[1, 3] = (lr_ * cr_ - lf_ * cf_) / mass_ / vx
A[2, 3] = 1.0
A[3, 1] = (lr_ * cr_ - lf_ * cf_) / iz_ / vx
A[3, 2] = (lf_ * cf_ - lr_ * cr_) / iz_
A[3, 3] = -1.0 * (lf_ * lf_ * cf_ + lr_ * lr_ * cr_) / iz_ / vx

B = np.zeros((4, 1), dtype=np.float32)
B[1, 0] = cf_ / mass_
B[3, 0] = lf_ * cf_ / iz_

x_0 = np.array([-0.5, 0, 0, 0])

Q = np.eye(4)
R = np.eye(1) * 10

control_limit = { 'lo': np.array([-0.261799]), 'up': np.array([0.261799])}


class Controller:
    def __init__(self, dt):
        self.lqr = LQR(A, B, Q, R)
        self.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.lqr.set_reference(ref)
        cin = self.lqr.update(feedback_value, current_time)
        return cin

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.lqr.set_control_limit(self.control_lo, self.control_up)

    def clear(self):
        pass


class LaneKeeping(Simulator):
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Lane Keeping ' + name, dt, max_index)
        self.linear(A, B)
        controller = Controller(dt)
        settings = {
            'init_state': x_0,
            'feedback_type': 'state',
            'controller': controller
        }
        if noise:
            settings['noise'] = noise
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 1000
    dt = 0.01
    ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.001}
        }
    }
    lk = LaneKeeping('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert lk.cur_index == i
        lk.update_current_ref(ref[i])
        # attack here
        lk.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in lk.refs[:max_index + 1]]
    y_arr = [x[0] for x in lk.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()
