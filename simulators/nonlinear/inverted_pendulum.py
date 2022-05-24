# Ref: Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control (Session 8.7, Page 300)

import numpy as np
from utils import Simulator, LQR

# parameters
m = 1  # mass of rob
M = 5  # mass of cart
L = 2  # length of rob
g = -10
d = 1  # dumping (friction)
b = 1  # pendulum up (b=1)


def inverted_pendulum(t, x, u, params={}):
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m * L * L * (M + m * (1 - Cx * Cx))

    dx = np.zeros((4,))
    dx[0] = x[1]
    dx[1] = (1 / D) * (-m * m * L * L * g * Cx * Sx + m * L * L * (m * L * x[3] * x[3] * Sx - d * x[1])) + m * L * L * (
            1 / D) * u
    dx[2] = x[3]
    dx[3] = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * (m * L * x[3] * x[3] * Sx - d * x[1])) - m * L * Cx * (
            1 / D) * u
    return dx


x_0 = np.array([-1, 0, np.pi + 0.1, 0])
control_limit = {
    'lo': np.array([-50]),
    'up': np.array([50])
}

# control parameters
A = np.array([[0, 1, 0, 0],
              [0, -d / M, b * m * g / M, 0],
              [0, 0, 0, 1],
              [0, -b * d / (M * L), -b * (m + M) * g / (M * L), 0]])
B = np.array([[0], [1 / M], [0], [b * 1 / (M * L)]])

R = np.array([[0.0001]])
Q = np.eye(4)


class Controller:
    def __init__(self, dt, control_limit=None):
        self.lqr = LQR(A, B, Q, R)
        self.lqr.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.lqr.set_reference(ref)
        cin = self.lqr.update(feedback_value, current_time)
        return cin


class InvertedPendulum(Simulator):
    """
    States: (4,)
        x[0]: location of cart
        x[1]: dx[0]
        x[2]: pendulum angle  (down:0, up:pi)
        x[3]: dx[1]
    Control Input: (1,)  [control_limit]
        u[0]: force on the cart
    Output: (4,)
        State Feedback
    Controller: LQR
    """

    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('Inverted Pendulum ' + name, dt, max_index)
        self.nonlinear(ode=inverted_pendulum, n=4, m=1, p=4)
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
    ref = [np.array([1, 0, np.pi, 0])] * (max_index+1)
    # bias attack example
    from utils import Attack
    bias = np.array([-1, 0, 0, 0])
    bias_attack = Attack('bias', bias, 300)
    ip = InvertedPendulum('test', dt, max_index)
    for i in range(0, max_index + 1):
        assert ip.cur_index == i
        ip.update_current_ref(ref[i])
        # attack here
        ip.cur_feedback = bias_attack.launch(ip.cur_feedback, ip.cur_index, ip.states)
        ip.evolve()
    # print results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1)
    ax1, ax2, ax3 = ax
    t_arr = np.linspace(0, 10, max_index + 1)
    ref1 = [x[0] for x in ip.refs[:max_index + 1]]
    y1_arr = [x[0] for x in ip.outputs[:max_index + 1]]
    ax1.set_title('x0-location')
    ax1.plot(t_arr, y1_arr, t_arr, ref1)
    ref2 = [x[2] for x in ip.refs[:max_index + 1]]
    y2_arr = [x[2] for x in ip.outputs[:max_index + 1]]
    ax2.set_title('x2-angle')
    ax2.plot(t_arr, y2_arr, t_arr, ref2)
    u_arr = [x[0] for x in ip.inputs[:max_index + 1]]
    ax3.set_title('u-force')
    ax3.plot(t_arr, u_arr)
    plt.show()
