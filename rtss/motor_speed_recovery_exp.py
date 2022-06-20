from simulators import MotorSpeed
import numpy as np
from utils.attack import Attack

from utils.formal import Zonotope
from utils.formal.reachability import ReachableSet
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.strip import Strip

# --------------- parameters  -----------------------------
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

x_0 = GaussianDistribution(np.array([4.4392062,  42.17318198]), np.zeros((2, 2)))
s = Strip(np.array([-1, 0]), a=-4.2, b=-3.8)
reach.init(x_0, s)

# fig_setting = {'x1': 0, 'x2': 80, 'y1': 0, 'y2': 90,
#                'strip': True, 'routine': True,
#                'zonotope': True, 'distribution': True}
fig_setting = {'x1': 3.5, 'x2': 4.5, 'y1': 30, 'y2': 48,
               'strip': False, 'routine': True,
               'zonotope': True, 'distribution': True,
               'head_width': 0.01, 'width': 0.002}
# X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(2)
# # reach.plot(X_k, D_k, alpha, fig_setting)
#
# X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(10)
# # reach.plot(X_k, D_k, alpha, fig_setting)
# print(P)
#
# i, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=0.95, max_k=40)
# print('i=', i, 'found=', satisfy, 'P=', P)
# reach.plot(X_k, D_k, alpha, fig_setting)

k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(10)
print(k, P, arrive)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(1)
print('k=', 1, '   P=', P, '   D_k=', D_k)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(2)
print('k=', 2, '   P=', P, '   D_k=', D_k)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(3)
print('k=', 3, '   P=', P, '   D_k=', D_k)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(4)
print('k=', 4, '   P=', P, '   D_k=', D_k)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(5)
print('k=', 5, '   P=', P, '   D_k=', D_k)


