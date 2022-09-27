import numpy as np
from scipy.signal import StateSpace

from lane_keeping_model import LaneKeeping
from utils.formal.zonotope import Zonotope
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.strip import Strip
from utils.controllers.MPC_OSQP import MPC

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

mpc_settings = {
    'Ad': sysd.A, 'Bd': sysd.B,
    'Q': np.eye(4), 'QN': np.eye(4), 'R': np.eye(1) * 10,
    'N': 50,
    # 'ddl': None, 'target_lo': , 'target_up':,
    # 'safe_lo': , 'safe_up':,
    'control_lo': np.array([-0.261799]), 'control_up': np.array([0.261799]),
    'ref': np.array([0, 0, 0, 0])
}

mpc = MPC(mpc_settings)
cin = mpc.update(feedback_value=np.array([-2.0269287 , -0.5748443 , -0.6124138 ,  0.07075875]))
print(cin)


