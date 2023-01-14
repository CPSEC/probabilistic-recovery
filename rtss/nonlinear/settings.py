import numpy as np
from simulators.nonlinear.continuous_stirred_tank_reactor import CSTR

from utils.attack import Attack
from utils.formal.strip import Strip


# ------------------ continuous_stirred_tank_reactor (CSTR) -------------------
class cstr:
    # needed by 0_attack_no_recovery
    name = 'cstr'
    max_index = 300
    dt = 0.1
    ref = [np.array([0.98189, 300.00013])] * (max_index+1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.array([[1e-2, 0], [0, 0.4]])}
        }
    }
    # noise = {}
    model = CSTR(name, dt, max_index, noise=noise)
    control_lo = np.array([250])
    control_up = np.array([350])
    model.controller.set_control_limit(control_lo, control_up)

    attack_start_index = 90  # index in time
    recovery_index = 105  # index in time
    bias = np.array([0, -30])
    unsafe_states_onehot = [0, 1]
    attack = Attack('bias', bias, attack_start_index)

    output_index = 1  # index in state
    ref_index = 1  # index in state

    safe_set_lo = np.array([-5, 250])
    safe_set_up = np.array([5, 330])
    target_set_lo = np.array([-5, 296])
    target_set_up = np.array([5, 304])
    control_lo = np.array([250])
    control_up = np.array([350])
    recovery_ref = np.array([0.98189, 300.00013])

    s = Strip(np.array([0, 1]), a=target_set_lo[output_index], b=target_set_up[output_index])
    k_given = 40

    # Q = np.diag([1, 1])
    # QN = np.diag([1, 1])
    Q = np.diag([1, 100])
    QN = np.diag([1, 100])
    R = np.diag([1])

    MPC_freq = 1
    nx = 2
    nu = 1

    # plot
    y_lim = (280, 340)
    x_lim = (8, dt * 117)
    strip = (target_set_lo[output_index], target_set_up[output_index])
    y_label = 'Temperature [K]'

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([274.57786])
    x_ss = np.array([0.98472896, 300.00335862])
    jh = lambda x, u: np.array([[1, 0]])

    # attack_start_index = 90
    # bias = np.array([0, -25])
    # attack = Attack('bias', bias, attack_start_index)
    # recovery_index = 100
    #
    # output_index = 1 # index in state
    # ref_index = 1 # index in state
    #
    # # needed by 1_recovery_given_p
    # s = Strip(np.array([-1, 0]), a=-4.2, b=-3.8)
    # P_given = 0.95
    # max_recovery_step = 40
    # # plot
    # y_lim = (280, 360)
    # x_lim = (8, dt * 200)
    # strip = (target_set_lo[output_index], target_set_up[output_index])
    # y_label = 'Temperature [K]'
    #
    # kf_C = np.array([[0, 1]])
    # k_given = 40  # new added
    # kf_R = np.diag([1e-7])
    #
    # # baseline
    # safe_set_lo = np.array([4, 30])
    # safe_set_up = np.array([5.07, 80])
    # target_set_lo = np.array([3.9, 35.81277766])
    # target_set_up = np.array([4.1, 60])
    # recovery_ref = np.array([4,  41.81277766])
    # Q = np.diag([100, 1])
    # QN = np.diag([100, 1])
    # R = np.diag([1])