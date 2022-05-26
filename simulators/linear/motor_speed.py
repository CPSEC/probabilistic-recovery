# Ref: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=ControlPID

import numpy as np

from utils import PID, Simulator

# system dynamics
J = 0.01   # moment of inertia of the rotor
b = 0.1    # motor viscous friction constant
Ke = 0.01  # electromotive force constant
Kt = 0.01  # motor torque constant
R = 1      # electric resistance
L = 0.5    # electric inductance

A = [[-b / J, Kt / J], [-Ke / L, -R / L]]
B = [[0], [1 / L]]
C = [[1, 0]]

x_0 = np.array([0.0, 0.0])

# control parameters
P = 19
I = 37
D = 0.1

control_limit = {
    'lo': np.array([0]),
    'up': np.array([60])
}


class Controller:
    def __init__(self, dt):
        self.dt = dt
        self.pid = PID(P, I, D, current_time=-dt)
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


class MotorSpeed(Simulator):
    """
            States: (2,)
                x[0]: the rotational speed of the shaft
                x[1]: electric current
            Control Input: (1,)
                u[0]: voltage source
            Output:  (1,)
                y[0]: the rotational speed of the shaft
                Output Feedback
            Controller: PID
            """
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Motor Speed ' + name, dt, max_index)
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
    ref = [np.array([5])] * 201 + [np.array([4])] * 200 + [np.array([5])] * 100
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.1}
        }
    }
    motor_speed = MotorSpeed('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert motor_speed.cur_index == i
        motor_speed.update_current_ref(ref[i])
        # attack here
        motor_speed.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in motor_speed.refs[:max_index + 1]]
    y_arr = [x[0] for x in motor_speed.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()

    u_arr = [x[0] for x in motor_speed.inputs[:max_index + 1]]
    plt.plot(t_arr, u_arr)
    plt.show()
