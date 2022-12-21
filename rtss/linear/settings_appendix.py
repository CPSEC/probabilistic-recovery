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
            'param': {'C': np.array([[0.03, 0], [0, 0.04]])}
        }
    }
    model = MotorSpeed('bias', dt, max_index, noise)
    control_lo = np.array([0])
    control_up = np.array([50])
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
    x_lim = (2.8, 4.2)
    y_lim = (3.65, 5.2)
    y_label = 'rotational speed - rad/sec'
    strip = (4.2, 3.8)

    kf_C = np.array([[0, 1]])
    k_given = 40  # new added
    kf_R = np.diag([1e-7])

    # baseline
    safe_set_lo = np.array([4, 30])
    safe_set_up = np.array([5.07, 80])
    target_set_lo = np.array([3.9, 35.81277766])
    target_set_up = np.array([4.1, 60])
    recovery_ref = np.array([4,  41.81277766])
    Q = np.diag([100, 1])
    QN = np.diag([100, 1])
    R = np.diag([1])


# -------------------- quadruple tank ----------------------------
class quadruple_tank_bias:
    # needed by 0_attack_no_recovery
    name = 'quadruple_tank_bias'
    max_index = 300
    dt = 1
    ref = [np.array([7, 7])] * 1001 + [np.array([7, 7])] * 1000
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.diag([0.05, 0.05, 0.05, 0.05])}
        }
    }
    model = QuadrupleTank('test', dt, max_index, noise)
    control_lo = np.array([0, 0])
    control_up = np.array([10, 10])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 150
    bias = np.array([-2.0, 0])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 160

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0]), a=-14.3, b=-13.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (140, 200)
    y_lim = (6.7, 9)
    y_label = 'water level - cm'
    strip = (7.15, 6.85)  # modify according to strip

    kf_C = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([0, 0, 0, 0])
    safe_set_up = np.array([20, 20, 20, 20])
    target_set_lo = np.array([13.8, 13.8, 0, 0])
    target_set_up = np.array([14.2, 14.2, 20, 20])
    recovery_ref = np.array([14, 14, 2, 2.5])
    Q = np.diag([1, 1, 0, 0])
    QN = np.diag([1, 1, 0, 0])
    R = np.diag([1, 1])


# -------------------- f16 ----------------------------
class f16_bias:
    # needed by 0_attack_no_recovery
    name = 'f16_bias'
    max_index = 550
    dt = 0.02
    ref = [np.array([0.0872665 * 57.3])] * 801
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.002}
        }
    }
    model = F16('test', dt, max_index, noise)
    control_lo = np.array([-25])
    control_up = np.array([25])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-10])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 470

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, 1, 0]), a=4.2/57.3, b=5.8/57.3)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (7.9, 8.95)
    y_lim = (4.5, 6.7)
    y_label = 'pitch angle - degree'
    strip = (5.1, 4.9)
    # strip = (8.76e-02 , 8.75e-02 )

    kf_C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([300, -5.01717652e-01, 0.04, -1])
    safe_set_up = np.array([5.02135552e+02,1.01717652e-01, 0.116, 1])
    # target_set_lo = np.array([13.8, 13.8, 0.0871 * 57.3, -20])
    # target_set_up = np.array([14.2, 14.2, 0.0874 * 57.3, 20])
    target_set_lo = np.array([3.52135331e+02, -1.51738001e-01,  0.0871,  3.05954232e-04])
    target_set_up = np.array([4.52135331e+02,  -0.51738001e-01, 0.0873,  4.05954232e-04])
    recovery_ref = np.array([4.02135552e+02, -1.01717652e-01, 0.0872665, 1.83150103e-04])
    Q = np.diag([1, 1, 10000, 10000])
    QN = np.diag([1, 1, 10000, 10000])
    R = np.diag([1])


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
            'param': {'C': np.diag([0.0003, 0.0002, 0.0003])}
        }
    }
    model = AircraftPitch('aircraft_pitch', dt, max_index, noise)
    control_lo = np.array([-20])
    control_up = np.array([20])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 500
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 550

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, -1]), a=-0.395, b=-0.005)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (9.7, 12.7)
    y_lim = (0, 1.3)
    y_label = 'pitch - rad'
    strip = (0.23, 0.17)

    kf_C = np.array([[1, 0, 0], [0, 1, 0]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([-5,    -5, 0])
    safe_set_up = np.array([5,    5, 1.175])
    target_set_lo = np.array([0.01624,    0.0, 0.19])
    target_set_up = np.array([0.05624,  0.01, 0.21])
    recovery_ref = np.array([0.05624,    0.00028221, 0.2])
    Q = np.diag([1, 100000, 100000])
    QN = np.diag([1, 100000, 100000])
    R = np.diag([1])


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
            'param': {'C': np.eye(5) * 0.02}
        }
    }
    model = Boeing('boeing747', dt, max_index, noise)
    control_lo = np.array([-30])
    control_up = np.array([30])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 430

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0, 0]), a=-1.4, b=-0.6)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (7.8,9.5)
    y_lim = (0.7, 2.0)
    y_label = 'Yaw angle - rad'
    strip = (0.85, 1.15)

    kf_C = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([0.0, -10, -5, -5, -30])
    #safe_set_up = np.array([1.817344443758, 5, 5, 50, 15])
    safe_set_up = np.array([1.819, 5, 20, 50, 25])
    target_set_lo = np.array([0.9, -10, -10, -10, -100])
    target_set_up = np.array([1.1, 10, 100, 100, 100])
    recovery_ref = np.array([1, 0, 0, 0, 0])
    Q = np.diag([100000, 1, 10000, 1, 1])
    QN = np.diag([100000, 1, 10000, 1, 1])
    R = np.diag([1]) * 0.1

def generate_list(dimension, value):
    temp = []
    for i in range(dimension):
        temp.append(value)
    return temp
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

    kf_C = np.zeros((44, 45))  # depend on attack
    for i in range(44):
        if i < y_point:
            kf_C[i][i] = 1
        elif i >= y_point:
            kf_C[i][i + 1] = 1
    k_given = 40
    temp = []
    for i in range(44):
        temp.append(1e-7)
    kf_R = np.diag(temp)

    # baseline
    temp = generate_list(45, -10)
    safe_set_lo = np.array(temp)

    temp = generate_list(45, 1000)
    safe_set_up = np.array(temp)

    temp = generate_list(45, -10)
    temp[y_point] = 14.8
    target_set_lo = np.array(temp)

    temp = generate_list(45, 1000)
    temp[y_point] = 15.2
    target_set_up = np.array(temp)

    temp = generate_list(45, 0)
    temp[y_point] = 15
    recovery_ref = np.array(temp)

    temp = generate_list(45, 1)
    temp[y_point] = 1
    Q = np.diag(temp)
    QN = np.diag(temp)
    R = np.diag([1]) * 10000





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
    control_lo = np.array([-5, -5, -5, -5])
    control_up = np.array([5, 5, 5, 5])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 450

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0, 0, 0, 0]), a=-2.3, b=-1.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (7.5,10)
    y_lim = (1.5, 2.7)
    y_label = 'relative distance error with car 1 and 2'
    strip = (1.9, 2.1)

    kf_C = np.array([[0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([1, 1, 1, 0, 0, 0, 0])
    safe_set_up = np.array([2.5, 3, 3, 5, 5, 5, 5])
    target_set_lo = np.array([1.95, 0, 0, 0, 0, 0, 0])
    target_set_up = np.array([2.05, 5, 5, 5, 5, 5, 5])
    recovery_ref = np.array([2, 2, 2, 2, 2, 2, 2])
    Q = np.diag([1, 1, 1, 1, 1, 1, 1])
    QN = np.diag([1, 1, 1, 1, 1, 1, 1])
    R = np.diag([1, 1, 1, 1])


# -------------------- rlc_circuit ----------------------------
class rlc_circuit_bias:
    # needed by 0_attack_no_recovery
    name = 'rlc_circuit'
    max_index = 800
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
    recovery_index = 415

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0]), a=-3.2, b=-2.8)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (7.9, 8.7)
    y_lim = (2.7, 4.1)
    y_label = 'Capacitor Voltage - V'
    strip = (2.8, 3.2)

    kf_C = np.array([[0, 1]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7])

    # baseline
    safe_set_lo = np.array([2.5, -1])
    safe_set_up = np.array([4, 0.5])
    target_set_lo = np.array([2.95, -1])
    target_set_up = np.array([3.05, 1])
    recovery_ref = np.array([3, -1.52863266e-05])
    Q = np.diag([1, 1])
    QN = np.diag([1, 1])
    R = np.diag([1])


# -------------------- quadrotor ----------------------------
class quadrotor_bias:
    # needed by 0_attack_no_recovery
    name = 'quadrotor'
    max_index = 1000
    dt = 0.02
    ref = [np.array([4])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(12) * 0.08}
        }
    }
    model = Quadrotor('quadrotor', dt, max_index, noise)
    control_lo = np.array([-50])
    control_up = np.array([50])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 400
    bias = np.array([-2])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 410

    # needed by 1_recovery_given_p
    s = Strip(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]), a=-4.5, b=-3.5)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 5
    x_lim = (7.9, 8.9)
    y_lim = (3.5, 6.2)
    y_label = 'Altitude - m'
    strip = (4.3, 3.7)

    # kf_C = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # depend on attack
    kf_C = np.zeros((11, 12))
    for i in range(11):
        kf_C[i][i] = 1
    # print(kf_C)
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([-1, -1, -1, -10, -10, -10, -10, -10, -10, -10, -10, 3.5])
    safe_set_up = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7.8])
    target_set_lo = np.array([-1, -1, -1, -10, -10, -10, -10, -10, -10, -10, -10, 3.9])
    target_set_up = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4.1])
    recovery_ref = np.array([5.67886532e-03,  5.25493934e-03, -1.59956078e-03,  1.21647831e-03,
  2.57637592e-04,  1.17194568e-03, -2.24887084e-01,  1.84248778e-01,
 -2.22306362e-03, -5.65196133e-01,  4.69398702e-01,  4])
    Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    QN = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    R = np.diag([1])


# -------------------- lane keeping ----------------------------
class lane_keeping:
    # needed by 0_attack_no_recovery
    name = 'lane_keeping_bias'
    max_index = 250
    dt = 0.05
    ref = [np.array([0, 0, 0, 0])] * (max_index + 1)
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.eye(4) * 0.002}
        }
    }
    model = LaneKeeping('test', dt, max_index, noise)
    control_lo = np.array([-0.261799])
    control_up = np.array([0.261799])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 100
    bias = np.array([-0.6, 0, 0, 0])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 130

    # needed by 1_recovery_given_p
    s = Strip(np.array([1, 0, 0, 0]), a=-0.05, b=0.05)
    P_given = 0.95
    max_recovery_step = 150
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (4.5, 9.3)
    y_lim = (-0.1, 0.6)
    y_label = 'lateral error - m'
    strip = (-0.05, 0.05)

    kf_C = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    k_given = 100
    kf_R = np.diag([1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([-0.03, -0.5, -0.5, -0.5])
    safe_set_up = np.array([0.55, 1, 1, 1])
    target_set_lo = np.array([-0.04, -0.5, -0.5, -0.5])
    target_set_up = np.array([0.04, 0.5, 0.5, 0.5])
    recovery_ref = np.array([0, 0.00726603,  0.00359684, -0.00704901])
    Q = np.diag([100, 1, 1, 1])
    QN = np.diag([100, 1, 1, 1])
    R = np.diag([1])
