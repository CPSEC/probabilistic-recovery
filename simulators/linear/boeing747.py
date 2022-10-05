#Ref: https://web.stanford.edu/class/archive/ee/ee392m/ee392m.1056/Lecture2_LinearSystems.pdf page 6
# Nenad Popovich. Lateral motion control of boeing 747
# by using full-order observer. In 2019 5th International
# Conference on Control, Automation and Robotics (IC-
# CAR), pages 377–383, 2019.
import numpy as np

from utils import PID, Simulator, LQRSSE, LQR

# system dynamics
A = np.array([[0, 0, 1, 0, 0],
              [0, -0.0558, -0.9968, 0.0802, 0.0415],
              [0, 0.598, -0.115, -0.0318, 0],
              [0, -3.05, 0.388, -0.4650, 0],
              [0, 0, 0.0805, 1, 0]])
B = np.array([[0], [0.00729], [-0.475], [0.153], [0]])
C = np.array([[1, 0, 0, 0, 0]])

x_0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])

# control parameters
R = np.array([[10]])
Q = np.eye(5)
# KP = 0
# KI = 0
# KD = 0
KP = -100
KI = 0
KD = 30
control_limit = {'lo': [-30], 'up': [30]}


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

# class Controller:
#     def __init__(self, dt, control_limit=None):
#         self.lqr = LQR(A, B, Q, R)
#         # self.lqr.set_control_limit(control_limit['lo'], control_limit['up'])
#
#     def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
#         self.lqr.set_reference(ref)
#         cin = self.lqr.update(feedback_value, current_time)
#         return cin

class Boeing(Simulator):
    """
              States: (5,)
                  x[0]: Yaw angle
                  x[1]: Side-slip angle
                  x[2]: Yaw rate
                  x[3]: roll rate
                  x[4]: roll angle
              Control Input: (1,)
                  u[0]: Rudder
              Output:  (2,)
                  y[0]: Yaw angle
                  Output Feedback
              Controller: PID
              """

    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Boeing' + name, dt, max_index)
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
    max_index = 2000
    dt = 0.02
    ref = [np.array([1])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(5) * 0.01}
        }
    }
    boeing = Boeing('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert boeing.cur_index == i
        boeing.update_current_ref(ref[i])
        # attack here
        boeing.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in boeing.refs[:max_index + 1]]
    y_arr = [x[0] for x in boeing.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)

    plt.show()

    # u_arr = [x[0] for x in boeing.inputs[:max_index + 1]]
    # plt.plot(t_arr, u_arr)
    # plt.show()
