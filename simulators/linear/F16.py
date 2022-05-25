# Ref: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling

import numpy as np
import math
from utils import PID, Simulator

# system dynamics
A = np.array([[-1.93*math.pow(10, -2), 8.82, -32.2, -0.48],
              [-2.54 * math.pow(10, -4), -1.02, 0, 0.91],
              [0, 0, 0, 1],
              [2.95 * math.pow(10, -12), 0.82, 0, -1.08]])
B = np.array([[0.17], [-2.15 * math.pow(10, -3)], [0], [-0.18]])
C = np.array([0, 0, 57.3, 0]).reshape((4,))
D = np.array([0.0])

x_0 = np.array([[0.0], [0.0], [0.0], [0.0]]).reshape((4,))

# control parameters
KP = -2
KI = 0
KD = 0.1


class Controller:
    def __init__(self, dt):
        self.pid = PID(KP, KI, KD, current_time=-dt)
        self.pid.clear()
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        # self.pid.setControlLimit()

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid.set_reference(ref[0])
        cin = self.pid.update(feedback_value[0], current_time)
        return np.array([cin])


class F16(Simulator):
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
    ref = [np.array([5])] * 201 + [np.array([4])] * 200 + [np.array([5])] * 100
    noise = {
        'measurement': {
            'type': 'white',
            'param': np.array([1]) * 0.05
        }
    }
    f16 = F16('test', dt, max_index, None)
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
