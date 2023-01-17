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
            'param': {'C': np.array([[1e-2, 0], [0, 0.5]])}
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
    target_set_lo = np.array([-5, 294])
    target_set_up = np.array([5, 306])
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
    y_lim = (290, 350)
    x_lim = (8.7, dt * 125)
    strip = (target_set_lo[output_index], target_set_up[output_index])
    y_label = 'Temperature [K]'

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([274.57786])
    x_ss = np.array([0.98472896, 300.00335862])
    jh = lambda x, u: np.array([[1, 0]])

    # date file
    state_names = ['Ca', 'T']
    legend_loc = 'upper right'
    y_lim_time = (0, 15)

# only for plot
class svl:
    name = 'svl'
    state_names = ['x', 'y', 'yaw']
    output_index = 1
    x_lim = (30, 38.5)
    y_lim = (44, 56)
    from math import pi
    target_set_lo = np.array([-55, 50.45-4.5, -pi])
    target_set_up = np.array([55, 50.45-3, pi])
    strip = (target_set_lo[output_index], target_set_up[output_index])
    dt = 0.05
    attack_start_index = 620
    recovery_index = 650
    y_label = 'y position [m]'
    legend_loc = 'lower left'

    y_lim_time = (0, 25)
    nx = 3
    s = Strip(np.array([0, 1, 0]), a=target_set_lo[output_index], b=target_set_up[output_index])
