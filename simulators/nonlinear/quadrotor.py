import numpy as np

from utils import PID, Simulator, LQRSSE, LQR

# system dynamics
g = 9.81
m = 0.468
A = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
B = [[0], [0], [0], [0], [0], [0], [0], [0], [1 / m], [0], [0], [0]]
C = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
D = [[0], [0], [0], [0], [0], [0]]

x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# control parameters
control_limit = {
    'lo': np.array([-50]),
    'up': np.array([50])
}

KP = 6
KI = 0
KD = 0.06


class Controller:
    def __init__(self, dt, control_limit=None):
        self.pid = PID(KP, KI, KD, current_time=-dt)
        self.pid.clear()
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        self.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid.set_reference(ref[0])
        cin = self.pid.update(feedback_value[5], current_time)
        return np.array([cin])

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.pid.set_control_limit(self.control_lo[0], self.control_up[0])


class Quadrotor(Simulator):
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Quadrotor ' + name, dt, max_index)
        self.linear(A, B, C)
        controller = Controller(dt, control_limit)
        settings = {
            'init_state': x_0,
            'feedback_type': 'output',
            'controller': controller
        }
        if noise:
            settings['noise'] = noise
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 600
    dt = 0.02
    ref = [np.array([2])] * 201 + [np.array([4])] * 200 + [np.array([2])] * 200
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(12) * 0.001}
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
    y_arr = [x[5] for x in quadrotor.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()
