# Ref: https://apps.dtic.mil/sti/pdfs/ADA339188.pdf page 26

import numpy as np
import math
from utils import PID, Simulator

# system dynamics
A = np.array([[-1.93 * math.pow(10, -2), 8.82, -32.2, -0.48],
              [-2.54 * math.pow(10, -4), -1.02, 0, 0.91],
              [0, 0, 0, 1],
              [2.95 * math.pow(10, -12), 0.82, 0, -1.08]])
B = np.array([[0.17], [-2.15 * math.pow(10, -3)], [0], [-0.18]])
C = np.array([0, 0, 57.3, 0]).reshape((4,))
D = np.array([0.0])

x_0 = np.array([[500.0], [0.0393], [0.0], [0.0393]]).reshape((4,))

# control parameters
# KP = -1
# KI = 1.1
# KD = 0.00594
KP = -1.5
KI = -0.5
KD = 0.2
control_limit = {'lo': [-25], 'up': [25]}
#Ref: https://archive.siam.org/books/dc11/f16/Model.pdf page 6

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


class F16(Simulator):
    """
            States: (4,)
                x[0]: velocity V[ft/sec]
                x[1]: angle of attack [rad]
                x[2]: pitch angle [rad]
                x[3]: pitch rate [rad/sec]
            Control Input: (1,)
                u[0]: elevator deflection [deg]
            Output:  (1,)
                y[0]: pitch angle * 57.3
                Output Feedback
            Controller: PID
            """

    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('F16 ' + name, dt, max_index)
        self.linear(A, B, C, D)
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
    dt = 0.02
    ref = [np.array([0.0872665 * 57.3])] * 501
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.0001}
        }
    }
    f16 = F16('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert f16.cur_index == i
        f16.update_current_ref(ref[i])
        # attack here
        f16.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in f16.refs[:max_index + 1]]
    y_arr = [x[0] for x in f16.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()

    # u_arr = [x[0] for x in f16.inputs[:max_index + 1]]
    # plt.plot(t_arr, u_arr)
    # plt.show()
