import numpy as np
from simulators.linear.motor_speed import MotorSpeed
from simulators.linear.quadruple_tank import QuadrupleTank
from utils.attack import Attack
from utils.formal.strip import Strip


# --------------------- motor speed -------------------
class motor_speed_bias:
    # needed by 0_attack_no_recovery
    name = 'motor_speed_bias'
    max_index = 500
    dt = 0.02
    ref = [np.array([4])] * 501
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.array([[0.03, 0], [0, 0.001]])}
        }
    }
    model = MotorSpeed('bias', dt, max_index, noise)
    control_lo = np.array([0])
    control_up = np.array([60])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 150
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 200

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0]), a=-4.3, b=-3.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0


# -------------------- quadruple_tank.py ----------------------------
class quadruple_tank_bias:
    # needed by 0_attack_no_recovery
    name = 'quadruple_tank_bias'
    max_index = 200
    dt = 1
    ref = [np.array([7, 7])] * 1001 + [np.array([7, 7])] * 1000
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.1}
        }
    }
    model = QuadrupleTank('test', dt, max_index, noise)
    control_lo = np.array([0, 0])
    control_up = np.array([10, 10])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 80
    bias = np.array([-2.0, 0])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 120

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0]), a=-15, b=-14.6)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
