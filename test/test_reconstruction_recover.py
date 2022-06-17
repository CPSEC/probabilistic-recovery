from time import perf_counter

import numpy as np
from simulators.linear.motor_speed import MotorSpeed
from utils.attack import Attack
from utils.formal.strip import Strip
import matplotlib.pyplot as plt
from utils.observers.kalman_filter import KalmanFilter
from utils.formal.zonotope import Zonotope
from utils.formal.reachability import ReachableSet
from utils.formal.gaussian_distribution import GaussianDistribution

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
    recovery_index = 220

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


exps = [motor_speed_bias]
given_P = 0.9
result = {}
for exp in exps:
    print('=' * 20, exp.name, '=' * 20)
    result[exp.name] = {}
    result[exp.name][given_P] = {}
    k = None
    rec_u = None
    recovery_complete_index = None

    u_lo = exp.model.controller.control_lo
    u_up = exp.model.controller.control_up
    U = Zonotope.from_box(u_lo, u_up)
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=100)

    A = exp.model.sysd.A
    B = exp.model.sysd.B
    C = exp.kf_C
    D = exp.model.sysd.D
    Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
    R = np.zeros((C.shape[0], C.shape[0]))
    kf = KalmanFilter(A, B, C, D, Q, R)
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        if i == exp.attack_start_index - 1:
            print('normal_state=', exp.model.cur_x)
        if i == exp.recovery_index:
            print('recovery_start_state=', exp.recovery_index)
            print('recovery_start_state=', exp.model.cur_x)
            us = exp.model.inputs[exp.attack_start_index:exp.recovery_index]
            ys = (C @ exp.model.states[exp.attack_start_index+1:exp.recovery_index+1].T).T
            x_0 = exp.model.states[exp.attack_start_index]
            x_res, P_res = kf.multi_steps(x_0, np.zeros_like(A), us, ys)
            x_0 = GaussianDistribution(x_res[-1], P_res[-1])
            print('x_0=', x_0)

            reach.init(x_0, exp.s)
            # k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=exp.P_given, max_k=40)
            tic = perf_counter()
            k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=given_P, max_k=50)
            toc = perf_counter() - tic
            result[exp.name][given_P]['time'] = toc * 1000
            recovery_complete_index = exp.recovery_index + k
            print('k=', k, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)
            result[exp.name][given_P]['k'] = k
            result[exp.name][given_P]['max_k'] = 50
            # result[exp.name][given_P]['P'] = P
            # result[exp.name][given_P]['success'] = satisfy
            print('recovery_complete_index=', recovery_complete_index)
            rec_u = U.alpha_to_control(alpha)
            print(rec_u)
        if exp.recovery_index <= i < recovery_complete_index:
            # print(i)
            rec_u_index = i - exp.recovery_index
            u = rec_u[rec_u_index]
            exp.model.evolve(u)
            print(exp.model.cur_x)
        else:
            exp.model.evolve()

    # plot
    t_arr = np.linspace(0, exp.dt * exp.max_index, exp.max_index + 1)
    # output
    for i in range(exp.model.p):
        ref = [x[i] for x in exp.model.refs[:exp.max_index + 1]]
        y_arr = [x[i] for x in exp.model.outputs[:exp.max_index + 1]]
        fig = plt.figure()
        plt.title(exp.name + ' y_' + str(i))
        plt.plot(t_arr, ref, color='black', linestyle='dashed')
        plt.plot(t_arr, y_arr)
        t_rec_arr = np.linspace(exp.dt * exp.attack_start_index, exp.dt * exp.recovery_index, (exp.recovery_index-exp.attack_start_index)+1)
        y_rec_arr = (exp.model.sysd.C @ x_res.T).T
        plt.plot(t_rec_arr, y_rec_arr)
        plt.show()
    # control input
    for i in range(exp.model.m):
        u_arr = [x[i] for x in exp.model.inputs[:exp.max_index + 1]]
        fig = plt.figure()
        plt.title(exp.name + ' u_' + str(i))
        plt.plot(t_arr, u_arr)
        plt.show()
