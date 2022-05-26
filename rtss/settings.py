import numpy as np
from simulators.linear.motor_speed import MotorSpeed
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
    s = Strip(np.array([-1, 0]), a=-4.2, b=-3.8)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
