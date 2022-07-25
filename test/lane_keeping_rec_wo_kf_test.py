import numpy as np
from scipy.signal import StateSpace

from lane_keeping_model import LaneKeeping
from utils.formal.zonotope import Zonotope
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.strip import Strip

control_lo = np.array([-0.261799])
control_up = np.array([0.261799])
U = Zonotope.from_box(control_lo, control_up)
C_noise = np.diag([0.01, 0.001, 0.001, 0.001])
W = GaussianDistribution.from_standard(np.zeros((4,)) * 0.01, C_noise)
control_interval = 0.05
max_recovery_step = 100
safe_set = Strip(np.array([1, 0, 0, 0]), a=-0.05, b=0.05)

steer_model = LaneKeeping(5)
sysc = StateSpace(steer_model.A, steer_model.B, steer_model.C, steer_model.D)
sysd = sysc.to_discrete(control_interval)
reach = ReachableSet(sysd.A, sysd.B, U, W, max_step=max_recovery_step+2)

reconstructed_state = np.array([-3.901158, -0.13633914, -0.13676511, -0.44502825])
x_cur_update = GaussianDistribution(reconstructed_state, np.eye(4) * 0.01)
reach.init(x_cur_update, safe_set)

k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=max_recovery_step)
print(f"k={k}, X_k={X_k}, P={P}")
recovery_control_sequence = U.alpha_to_control(alpha)
print('recovery_control=', recovery_control_sequence[:10, :])

