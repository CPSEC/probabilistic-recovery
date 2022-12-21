import numpy as np
from simulators.linear.motor_speed import MotorSpeed
from simulators.linear.quadruple_tank import QuadrupleTank
from simulators.linear.F16 import F16
from simulators.linear.aircraft_pitch import AircraftPitch
from simulators.linear.boeing747 import Boeing
from simulators.linear.heat import Heat
from simulators.linear.platoon import Platoon
from simulators.linear.rlc_circuit import RlcCircuit
from simulators.linear.quadrotor import Quadrotor
from simulators.linear.lane_keeping import LaneKeeping

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
            'param': {'C': np.array([[0.1, 0], [0, 0.4]])}
        }
    }
    model = MotorSpeed('bias', dt, max_index, noise)
    control_lo = np.array([0])
    control_up = np.array([60])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 150
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 180

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0]), a=-4.3, b=-3.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = 2.8
    y_lim = (3.65, 5.5)
    y_label = 'rotational speed - rad/sec'
    strip = (4.3, 3.7)

    kf_C = np.array([[0, 1]])
    k_given = 40    #  new added

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
    s = Strip(np.array([-1, 0, 0, 0]), a=-14.6, b=-13.4)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = 75
    y_lim = (6, 10)
    y_label = 'water level - cm'
    strip = (7.3, 6.7)

    kf_C = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    k_given = 40
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
    control_lo = np.array([-25])
    control_up = np.array([25])
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
    x_lim = 7.5
    y_lim = (4.5, 7)
    y_label = 'pitch angle - rad'
    strip = (8.76e-02*57.3, 8.75e-02*57.3)

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
    attack_start_index = 500
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 700

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, -1]), a=-0.33, b=--0.27)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = 8
    y_lim = (0, 1.2)
    y_label = 'pitch - rad'
    strip = (0.33, 0.27)


# -------------------- boeing747 ----------------------------
class boeing747_bias:
    # needed by 0_attack_no_recovery
    name = 'boeing747'
    max_index = 800
    dt = 0.02
    ref = [np.array([1])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(5) * 0.0001}
        }
    }
    model = Boeing('boeing747', dt, max_index, noise)
    control_lo = np.array([-30])
    control_up = np.array([30])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 500

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0, 0]), a=-1.3, b=-0.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = 7.5
    y_lim = (0.4, 2.2)
    y_label = 'Yaw angle - rad'
    strip = (0.7, 1.3)

# -------------------- heat ----------------------------
class heat_bias:
    # needed by 0_attack_no_recovery
    name = 'heat'
    max_index = 1000
    dt = 1
    ref = [np.array([15])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(45) * 0.0001}
        }
    }
    model = Heat('heat', dt, max_index, noise)
    control_lo = np.array([-0.5])
    control_up = np.array([50])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 500

    # needed by 1_recovery_given_p
    l = np.zeros((45,))
    y_point = (45 + 1) // 3 * 2 - 1
    l[y_point] = -1
    s = Strip(l, a=-15.3, b=-14.7)
    P_given = 0.95
    max_recovery_step = 60
    # plot
    ref_index = 0
    output_index = 0

# -------------------- platoon ----------------------------
class platoon_bias:
    # needed by 0_attack_no_recovery
    name = 'platoon'
    max_index = 800
    dt = 0.02
    ref = [np.array([2])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(7) * 0.0001}
        }
    }
    model = Platoon('Platoon', dt, max_index, noise)
    control_lo = np.array([-5])
    control_up = np.array([5])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 500

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0, 0, 0, 0]), a=-2.3, b=-1.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0


# -------------------- rlc_circuit ----------------------------
class rlc_circuit_bias:
    # needed by 0_attack_no_recovery
    name = 'rlc_circuit'
    max_index = 600
    dt = 0.02
    ref = [np.array([3])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(2) * 0.0001}
        }
    }
    model = RlcCircuit('rlc_circuit', dt, max_index, noise)
    control_lo = np.array([-15])
    control_up = np.array([15])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 500

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0]), a=-3.3, b=-2.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = 7.7
    y_lim = (2.6, 4.1)
    y_label = 'Capacitor Voltage - V'
    strip = (2.7, 3.3)

# -------------------- quadrotor ----------------------------
class quadrotor_bias:
    # needed by 0_attack_no_recovery
    name = 'quadrotor'
    max_index = 600
    dt = 0.02
    ref = [np.array([4])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(12) * 0.0001}
        }
    }
    model = Quadrotor('quadrotor', dt, max_index, noise)
    control_lo = np.array([-50])
    control_up = np.array([50])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 500

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]), a=-4.3, b=-3.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 5
    x_lim = 7.7
    y_lim = (3.6, 5.2)
    y_label = 'Altitude - m'
    strip = (4.3, 3.7)


# -------------------- lane keeping ----------------------------
class lane_keeping:
    # needed by 0_attack_no_recovery
    name = 'lane_keeping_bias'
    max_index = 1000
    dt = 0.05
    ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.001}
        }
    }
    model = LaneKeeping('test', dt, max_index, noise)
    control_lo = np.array([-0.261799])
    control_up = np.array([0.261799])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 100
    bias = np.array([4, 0, 0, 0])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 130

    # needed by 1_recovery_given_p
    s = Strip(np.array([1, 0, 0, 0]), a=-0.05, b=0.05)
    P_given = 0.95
    max_recovery_step = 150
    # plot
    ref_index = 0
    output_index = 0
    x_lim = 4
    y_lim = (-4, 0.5)
    y_label = 'lateral error - m'
    strip = (-0.05, 0.05)

    kf_C = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    k_given = 100