import numpy as np
from simulators.linear.motor_speed import MotorSpeed
from simulators.linear.quadruple_tank import QuadrupleTank
from simulators.linear.F16 import F16
from simulators.linear.aircraft_pitch import AircraftPitch
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


# -------------------- quadruple tank ----------------------------
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


# -------------------- f16 ----------------------------
class f16_bias:
    # needed by 0_attack_no_recovery
    name = 'f16_bias'
    max_index = 800
    dt = 0.02
    ref = [np.array([0.0872665 * 57.3])] * 801
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.00001}
        }
    }
    model = F16('test', dt, max_index, noise)
    control_lo = np.array([-2])
    control_up = np.array([2])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 600

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, -1, 0]), a=-8.76e-02, b=8.75e-02)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0

# -------------------- aircraft_pitch ----------------------------
class aircraft_pitch_bias:
 # needed by 0_attack_no_recovery
    name = 'aircraft_pitch_bias'
    max_index = 1500
    dt = 0.02
    ref = [np.array([0.2])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(3) * 0.00001}
        }
    }
    model = AircraftPitch('aircraft_pitch', dt, max_index, noise)
    control_lo = np.array([-20])
    control_up = np.array([20])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 600
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 800

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, -1]), a=-0.33, b=--0.27)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0