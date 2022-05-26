#Ref:

import numpy as np
import math
from utils import PID, Simulator

# system dynamics
state_num = 45
A = np.zeros((state_num, state_num))
tmp = np.array([[1], [-2], [1]])
A[0:2, 0:1] = tmp[1:3, 0:1]
for i in range(state_num - 2):
    A[i:i + 3, i + 1:i + 2] = tmp
A[state_num - 2:state_num, state_num - 1:state_num] = tmp[0:2, 0:1]

if state_num == 2:
    A = np.array([[-2, 1], [1, -2]])

B = np.zeros((state_num, 1))
u_point = (state_num + 1) // 3 - 1
B[u_point, 0] = 1

C = np.zeros((1, state_num))
y_point = (state_num + 1) // 3 * 2 - 1
C[0, y_point] = 1

D = np.zeros((1, 1))

x_0 = np.zeros((state_num))

# control parameters
# KP = 0.00000099 * state_num * state_num * state_num * state_num
# KI = 0
# KD = 0
KP = 0.00000099 * state_num * state_num * state_num * state_num
KI = 0
KD = 0
control_limit = {'lo': [-0.5], 'up': [50]}


class Controller:
    def __init__(self, dt):
        self.dt = dt
        self.pid = PID(KP, KI, KD, current_time=-dt)
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        self.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid.set_reference(ref[0])
        cin = self.pid.update(feedback_value[0], current_time)
        return np.array([cin])

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.pid.set_control_limit(self.control_lo[0], self.control_up[0])

    def clear(self):
        self.pid.clear(current_time=-self.dt)


class Heat(Simulator):
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Aircraft Pitch ' + name, dt, max_index)
        self.linear(A, B, C)
        controller = Controller(dt)
        settings = {
            'init_state': x_0,
            'feedback_type': 'output',
            'controller': controller
        }
        if noise:
            settings['noise'] = noise
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 500
    dt = 0.2
    ref = [np.array([15])] * 201 + [np.array([15])] * 200 + [np.array([15])] * 100
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(200) * 0.01}
        }
    }
    heat = Heat('test', dt, max_index, None)
    for i in range(0, max_index + 1):
        assert heat.cur_index == i
        heat.update_current_ref(ref[i])
        # attack here
        heat.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in heat.refs[:max_index + 1]]
    y_arr = [x[0] for x in heat.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()
