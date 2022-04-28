import numpy as np

from utils import PID, Simulator

# system dynamics
J = 0.01
b = 0.1
Ke = 0.01
Kt = 0.01
R = 1
L = 0.5

A = [[-b / J, Kt / J], [-Ke / L, -R / L]]
B = [[0], [1 / L]]
C = [1, 0]

x_0 = np.array([0, 0])

# control parameters
P = 19
I = 35
D = 0.5


class Controller:
    def __init__(self, dt):
        self.pid = PID(P, I, D, current_time=-dt)
        self.pid.clear()
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        # self.pid.setControlLimit()

    def update(self, ref, feedback_value, current_time):
        self.pid.SetPoint = ref[0]
        self.pid.update(feedback_value[0], current_time)
        return np.array([self.pid.output])


class MotorSpeed(Simulator):
    def __init__(self, name, dt, max_index):
        super().__init__('Motor Speed ' + name, dt, max_index)
        self.linear(A, B, C)
        controller = Controller(dt)
        settings = {
            'init_state': x_0,
            'feedback_type': 'output',
            'controller': controller
        }
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 500
    dt = 0.02
    ref = [np.array([5])] * 201 + [np.array([4])] * 200 + [np.array([5])] * 100
    motor_speed = MotorSpeed('test', dt, max_index)
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
