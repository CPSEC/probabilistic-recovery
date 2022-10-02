#Ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9230365&casa_token=paW6MkN65vUAAAAA:JFJZQ3nC7fJ-6evzYq8DGWNrRAfd6qXUO
import numpy as np

from utils import PID, Simulator, LQRSSE, LQR

# system dynamics
kp = 2     # proportional gains of an on-board PD controller
kd = 1.5   # derivative gains of an on-board PD controller
beta = -0.1 # characterizes the loss of velocity as a result of friction
d_star = 2 # desired distance
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
R = np.eye(4) * 0.0008
Q = np.eye(7) * 1.2

control_limit = {
    'lo': np.array([-5]),
    'up': np.array([5])
}
class Controller:
    def __init__(self, dt, control_limit=None):
        self.lqr = LQR(A, B, Q, R)
        self.set_control_limit(control_lo=control_limit['lo'], control_up=control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.lqr.set_reference(ref)
        cin = self.lqr.update(feedback_value, current_time)
        return cin
    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.lqr.set_control_limit(self.control_lo[0], self.control_up[0])

    def clear(self):
        self.lqr.clear()

class Platoon(Simulator):
    """
              States: (7,)
                  x[0]: e12 relative distance error with car 1 and 2
                  x[1]: e23 relative distance error with car 2 and 3
                  x[2]: e34 relative distance error with car 3 and 4
                  x[3]: velocity of car 1
                  x[4]: velocity of car 2
                  x[5]: velocity of car 3
                  x[6]: velocity of car 4
              Control Input: (4,)
                  u[0]: acceleration of car 1
                  u[1]: acceleration of car 2
                  u[2]: acceleration of car 3
                  u[3]: acceleration of car 4
                  State Feedback
              Controller: PID
              """
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Platoon' + name, dt, max_index)
        self.linear(A, B)
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
