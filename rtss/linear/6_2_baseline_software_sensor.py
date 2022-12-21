from time import perf_counter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.controllers.MPC_OSQP import MPC
from utils.observers.full_state_bound import Estimator
from utils.controllers.MPC_OSQP import MPC


def in_target_set(target_lo, target_hi, x_cur):
    res = True
    for i in range(len(x_cur)):
        if target_lo[i] > x_cur[i] or target_hi[i] < x_cur[i]:
            res = False
            break
    return res


# ready exp: lane_keeping,
exps = [quadruple_tank_bias]
result = {}   # for print or plot
for exp in exps:

    # ----------------------------------- w/ kalman filter ---------------------------
    exp.model.reset()
    print('=' * 20, exp.name, '=' * 20)
    k = None
    rec_u = None
    recovery_complete_index = 10000
    x_cur_predict = None
    x_cur_update = None
    cin = None

    u_lo = exp.model.controller.control_lo
    u_up = exp.model.controller.control_up
    U = Zonotope.from_box(u_lo, u_up)
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=100)

    # C = exp.kf_C
    # D = exp.model.sysd.D
    # Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
    # R = np.zeros((C.shape[0], C.shape[0]))
    # kf = KalmanFilter(A, B, C, D, Q, R)
    est = Estimator(A, B, max_k=150, epsilon=1e-7)


    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        if i == exp.attack_start_index - 1:
            print('normal_state=', exp.model.cur_x)
        if i == exp.recovery_index:
            # x_0 estimation
            us = exp.model.inputs[exp.attack_start_index:exp.recovery_index]
            x_0 = exp.model.states[exp.attack_start_index]
            x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
            print(f'{exp.attack_start_index=},{exp.recovery_index=},\n{us=}')
            print(f'{x_cur_lo=},\n {x_cur_up=},\n {exp.model.states[i]=}')
            exp.model.cur_feedback = exp.model.sysd.C @ x_cur
            print(f'{i=}, {exp.model.cur_x=}')

        if exp.recovery_index < i <= recovery_complete_index:
            # check if it is in target set
            if in_target_set(exp.target_set_lo, exp.target_set_up, exp.model.cur_x):
                recovery_complete_index = i
                print('state after recovery:', exp.model.cur_x)
                step = recovery_complete_index - exp.recovery_index
                print(f'use {step} steps to recover.')
            else:
                us = [exp.model.inputs[i-1]]
                x_0 = exp.model.states[i-1]
                x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                exp.model.cur_feedback = exp.model.sysd.C @ x_cur
                print(f'{i=}, {exp.model.cur_x=}')

        exp.model.evolve()

    result['w/'] = {}
    result['w/']['outputs'] = deepcopy(exp.model.outputs)
    result['w/']['recovery_complete_index'] = recovery_complete_index
    # result['w/']['k'] = recovery_complete_index - exp.recovery_index
    # P_final = D_k.prob_in_strip(exp.s)
    # result['w/']['P_final'] = P_final

    # # plot
    # t_arr = np.linspace(0, exp.dt * exp.max_index, exp.max_index + 1)
    # # output
    # for i in range(exp.model.p):
    #     ref = [x[i] for x in exp.model.refs[:exp.max_index + 1]]
    #     y_arr = [x[i] for x in exp.model.outputs[:exp.max_index + 1]]
    #     fig = plt.figure()
    #     plt.title(exp.name + ' y_' + str(i))
    #     plt.plot(t_arr, ref, color='black', linestyle='dashed')
    #     plt.plot(t_arr, y_arr)
    #     # t_rec_arr = np.linspace(exp.dt * exp.attack_start_index, exp.dt * exp.recovery_index, (exp.recovery_index-exp.attack_start_index)+1)
    #     # y_rec_arr = (exp.model.sysd.C @ x_res.T).T
    #     # plt.plot(t_rec_arr, y_rec_arr)
    #     plt.show()
    # # control input
    # for i in range(exp.model.m):
    #     u_arr = [x[i] for x in exp.model.inputs[:exp.max_index + 1]]
    #     fig = plt.figure()
    #     plt.title(exp.name + ' u_' + str(i))
    #     plt.plot(t_arr, u_arr)
    #     plt.show()

    # print('='*20)
    # print('P_final_w/o_kf={} k_w/o_kf={}\nP_final_w/_kf={} k_w/_kf={}'
    #       .format(result['w/o']['P_final'], result['w/o']['k'], result['w/']['P_final'], result['w/']['k']))

    # plot
    plt.rcParams.update({'font.size': 18})  # front size
    fig = plt.figure(figsize=(8, 4))
    plt.title(exp.name+' y_'+str(exp.output_index))
    # recovery_complete_index = result['w/o']['recovery_complete_index']
    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # y_arr = [x[exp.output_index] for x in result['w/o']['outputs'][:recovery_complete_index + 1]]
    ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
    plt.plot(t_arr, ref, color='black', linestyle='dashed')
    # plt.plot(t_arr, y_arr, label='w/o')

    recovery_complete_index = result['w/']['recovery_complete_index']
    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    y_arr = [x[exp.output_index] for x in result['w/']['outputs'][:recovery_complete_index + 1]]
    plt.plot(t_arr, y_arr, label='w/')

    plt.vlines(exp.attack_start_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed')
    plt.vlines(exp.recovery_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted', linewidth=2.5)
    plt.hlines(exp.strip[0], exp.x_lim, recovery_complete_index*exp.dt, colors='grey')
    plt.hlines(exp.strip[1], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    plt.ylim(exp.y_lim)
    plt.xlim(exp.x_lim, recovery_complete_index*exp.dt)
    plt.legend(loc='best')
    # plt.savefig('./fig/'+exp.name+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # fig = plt.figure(figsize=(8, 4))
    # plt.title(exp.name + ' y_' + str(2))
    # # recovery_complete_index = result['w/o']['recovery_complete_index']
    # t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # # y_arr = [x[exp.output_index] for x in result['w/o']['outputs'][:recovery_complete_index + 1]]
    # ref = [x[2] for x in exp.model.refs[:recovery_complete_index + 1]]
    # plt.plot(t_arr, ref, color='black', linestyle='dashed')
    # # plt.plot(t_arr, y_arr, label='w/o')
    #
    # recovery_complete_index = result['w/']['recovery_complete_index']
    # t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # y_arr = [x[2] for x in result['w/']['outputs'][:recovery_complete_index + 1]]
    # plt.plot(t_arr, y_arr, label='w/')
    #
    # plt.vlines(exp.attack_start_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed')
    # plt.vlines(exp.recovery_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted',
    #            linewidth=2.5)
    # plt.hlines(exp.strip[0], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    # plt.hlines(exp.strip[1], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    # plt.ylim(exp.y_lim)
    # plt.xlim(exp.x_lim, recovery_complete_index * exp.dt)
    # plt.legend(loc='best')
    # plt.show()
    # output

