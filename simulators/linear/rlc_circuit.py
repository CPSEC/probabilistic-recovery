import numpy as np

from utils import PID, Simulator

# system dynamics
R = 10000
L = 0.5
C = 0.0001

A = [[0, 1 / C], [-1 / L, -R / L]]
B = [[0], [1 / L]]
C = [[1, 0]]
D = [[0]]

x_0 = np.array([0.0, 0.0])

# control parameters
KP = 5
KI = 5
KD = 0
control_limit = {'lo': [-15], 'up': [15]}


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


class RlcCircuit(Simulator):
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
    dt = 0.02
    ref = [np.array([2])] * 201 + [np.array([3])] * 200 + [np.array([2])] * 100
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.01}
        }
    }
    rlc_circuit = RlcCircuit('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert rlc_circuit.cur_index == i
        rlc_circuit.update_current_ref(ref[i])
        # attack here
        rlc_circuit.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in rlc_circuit.refs[:max_index + 1]]
    y_arr = [x[0] for x in rlc_circuit.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()
