from simulators import MotorSpeed
import numpy as np
from utils.attack import Attack

from utils.formal import Zonotope
from utils.formal.reachability import ReachableSet
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.strip import Strip

# --------------- parameters  -----------------------------
np.random.seed(0)
max_index = 500
dt = 0.02
ref = [np.array([4])] * 501
noise = {
    'process': {
        'type': 'white',
        'param': {'C': np.array([[0.08, 0], [0, 0.001]])}
    }
}
motor_speed = MotorSpeed('test', dt, max_index, noise)
attack_start_index = 150
bias = np.array([-1.0])
bias_attack = Attack('bias', bias, attack_start_index)
recovery_index = 200
# --------------- end of parameters -------------------------

u_lo = motor_speed.controller.control_lo
u_up = motor_speed.controller.control_up
U = Zonotope.from_box(u_lo, u_up)
print(U)

A = motor_speed.sysd.A
B = motor_speed.sysd.B
W = motor_speed.p_noise_dist
reach = ReachableSet(A, B, U, W, max_step=50)

x_0 = GaussianDistribution(np.array([5.09613504, 48.77378581]), np.zeros((2, 2)))
s = Strip(np.array([-1, 0]), a=-4.2, b=-3.8)
reach.init(x_0, s)

# fig_setting = {'x1': 0, 'x2': 80, 'y1': 0, 'y2': 90,
#                'strip': True, 'routine': True,
#                'zonotope': True, 'distribution': True}
fig_setting = {
               'strip': False, 'routine': False,
               'zonotope': True, 'distribution': False}
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(1)
reach.plot(X_k, D_k, alpha, fig_setting)

