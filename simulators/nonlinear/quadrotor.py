import numpy as np

from utils import PID, Simulator, LQRSSE, LQR

# system dynamics
g = 9.81
m = 0.468
A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
B = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1 / m], [0], [0], [0]])
C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# control parameters
control_limit = {
    'lo': np.array([-50]),
    'up': np.array([50])
}
R = np.array([0.001])
Q = np.eye(12)


class Controller:
    def __init__(self, dt, control_limit=None):
        self.lqr = LQR(A, B, Q, R)
        self.lqr.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.lqr.set_reference(ref)
        cin = self.lqr.update(feedback_value, current_time)
        return cin


class Quadrotor(Simulator):
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Quadrotor ' + name, dt, max_index)
        self.linear(A, B, C)
        controller = Controller(dt, control_limit)
        settings = {
            'init_state': x_0,
            'feedback_type': 'state',
            'controller': controller
        }
        if noise:
            settings['noise'] = noise
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 600
    dt = 0.02
    ref = [np.array([[2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])] * (max_index + 1)
    noise = {
        'measurement': {
            'type': 'white',
            'param': np.array([1]) * 0.05
        }
    }
    quadrotor = Quadrotor('test', dt, max_index, None)
    for i in range(0, max_index + 1):
        assert quadrotor.cur_index == i
        quadrotor.update_current_ref(ref[i])
        # attack here
        quadrotor.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in quadrotor.refs[:max_index + 1]]
    y_arr = [x[0] for x in quadrotor.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()
