import numpy as np

from utils import PID, Simulator, LQRSSE, LQR

# system dynamics
kp = 2
kd = 1.5
beta = -0.1
d_star = 2
A = np.array([[0, 0, 0, -1, 1, 0, 0],
              [0, 0, 0, 0, -1, 1, 0],
              [0, 0, 0, 0, 0, -1, 1],
              [kp, 0, 0, beta - kd, kd, 0, 0],
              [-kp, kp, 0, kd, beta - 2 * kd, kd, 0],
              [0, -kp, kp, 0, kd, beta - 2 * kd, kd],
              [0, 0, -kp, 0, 0, kd, beta - kd]])

B = np.concatenate((np.zeros((4, 3)), np.eye(4)), axis=1).T

x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# control parameters
R = np.eye(4) * 0.001
Q = np.eye(7)


class Controller:
    def __init__(self, dt, control_limit=None):
        self.lqr = LQR(A, B, Q, R)
        # self.lqr.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.lqr.set_reference(ref)
        cin = self.lqr.update(feedback_value, current_time)
        return cin


class Platoon(Simulator):

    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Platoon' + name, dt, max_index)
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
    max_index = 800
    dt = 0.02
    ref = [np.array([1])] * 301 + [np.array([2])] * 300 + [np.array([1])] * 200
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(7) * 0.01}
        }
    }
    platoon = Platoon('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert platoon.cur_index == i
        platoon.update_current_ref(ref[i])
        # attack here
        platoon.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in platoon.refs[:max_index + 1]]
    y_arr = [x[0] for x in platoon.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()
