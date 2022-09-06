from time import perf_counter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from settings_kf import motor_speed_bias, quadruple_tank_bias, lane_keeping
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter

exps = [motor_speed_bias]
exps = [quadruple_tank_bias]
exps = [lane_keeping]
result = {}   # for print or plot
for exp in exps:
    # ----------------------------------- w/o kalman filter ---------------------------
    print('=' * 20, exp.name, ' w/o kalman filter', '=' * 20)
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
            x_0 = GaussianDistribution(x_0, np.zeros((exp.model.n, exp.model.n)))
            reach.init(x_0, exp.s)
            x_res_point = reach.state_reconstruction(us)
            print('x_0=', x_res_point)

            reach.init(x_res_point, exp.s)
            # k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=exp.P_given, max_k=40)
            # k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=given_P, max_k=50)
            k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
            recovery_complete_index = exp.recovery_index + k
            print('k=', k, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)
            print('D_k=', D_k)
            print('recovery_complete_index=', recovery_complete_index)
            rec_u = U.alpha_to_control(alpha)
            print(rec_u)
        if exp.recovery_index <= i < recovery_complete_index:
            print(i)
            print(exp.model.cur_x)
            rec_u_index = i - exp.recovery_index
            u = rec_u[rec_u_index]
            exp.model.evolve(u)
        else:
            if i == recovery_complete_index:
                print('state after recovery:', exp.model.cur_x)
            exp.model.evolve()

    result['w/o'] = {}
    result['w/o']['outputs'] = deepcopy(exp.model.outputs)
    result['w/o']['recovery_complete_index'] = recovery_complete_index
    result['w/o']['k'] = k
    P_final = D_k.prob_in_strip(exp.s)
    result['w/o']['P_final'] = P_final

    # ----------------------------------- w/ kalman filter ---------------------------
    exp.model.reset()
    print('=' * 20, exp.name, '=' * 20)
    k = None
    rec_u = np.zeros((200, exp.model.m))
    recovery_complete_index = None
    x_cur_predict = None
    x_cur_update = None

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
            print("-" * 15, i, "-" * 15)
            us = exp.model.inputs[exp.attack_start_index:exp.recovery_index]
            ys = (C @ exp.model.states[exp.attack_start_index + 1:exp.recovery_index + 1].T).T
            x_0 = exp.model.states[exp.attack_start_index]
            x_res, P_res = kf.multi_steps(x_0, np.zeros_like(A), us, ys)
            x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
            print('x_cur_update=', x_cur_update)
            print('x_cur_real=', exp.model.cur_x)

            reach.init(x_cur_update, exp.s)
            k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
            recovery_complete_index = exp.recovery_index + k
            rec_u_temp = U.alpha_to_control(alpha)
            rec_u_index = i - exp.recovery_index
            rec_u[rec_u_index] = rec_u_temp[0]
            # print(rec_u)
        if exp.recovery_index < i < recovery_complete_index:
            print("-" * 15, i, "-" * 15)
            x_cur_predict = GaussianDistribution(*kf.predict(x_cur_update.miu, x_cur_update.sigma, exp.model.cur_u))
            y = C @ exp.model.cur_x
            x_cur_update = GaussianDistribution(*kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
            print('x_cur_update=', x_cur_update)
            print('x_cur_real=', exp.model.cur_x)

            reach.init(x_cur_update, exp.s)
            k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
            recovery_complete_index = i + k
            print('k=', k, ' D_k=', D_k)
            rec_u_temp = U.alpha_to_control(alpha)
            rec_u_index = i - exp.recovery_index
            rec_u[rec_u_index] = rec_u_temp[0]
            print('u=', rec_u_temp[0])

        if exp.recovery_index <= i < recovery_complete_index:
            rec_u_index = i - exp.recovery_index
            u = rec_u[rec_u_index]
            exp.model.evolve(u)
            print(exp.model.cur_x)
        else:
            if i == recovery_complete_index:
                print('state after recovery:', exp.model.cur_x)
            exp.model.evolve()

    result['w/'] = {}
    result['w/']['outputs'] = deepcopy(exp.model.outputs)
    result['w/']['recovery_complete_index'] = recovery_complete_index
    result['w/']['k'] = recovery_complete_index - exp.recovery_index
    P_final = D_k.prob_in_strip(exp.s)
    result['w/']['P_final'] = P_final

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
    #
    # # plot
    # plt.rcParams.update({'font.size': 18})  # front size
    # fig = plt.figure(figsize=(8, 4))
    # plt.title(exp.name+' y_'+str(exp.output_index))
    # recovery_complete_index = result['w/o']['recovery_complete_index']
    # t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # y_arr = [x[exp.output_index] for x in result['w/o']['outputs'][:recovery_complete_index + 1]]
    # ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
    # plt.plot(t_arr, ref, color='black', linestyle='dashed')
    # plt.plot(t_arr, y_arr, label='w/o')
    #
    # recovery_complete_index = result['w/']['recovery_complete_index']
    # t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # y_arr = [x[exp.output_index] for x in result['w/']['outputs'][:recovery_complete_index + 1]]
    # plt.plot(t_arr, y_arr, label='w/')
    #
    # plt.vlines(exp.attack_start_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed')
    # plt.vlines(exp.recovery_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted', linewidth=2.5)
    # plt.hlines(exp.strip[0], exp.x_lim, recovery_complete_index*exp.dt, colors='grey')
    # plt.hlines(exp.strip[1], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    # plt.ylim(exp.y_lim)
    # plt.xlim(exp.x_lim, recovery_complete_index*exp.dt)
    # plt.legend(loc='best')
    # # plt.savefig('./fig/'+exp.name+'.pdf', format='pdf', bbox_inches='tight')
    # plt.show()
    # # output

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

    fig = plt.figure(figsize=(8, 4))
    plt.title(exp.name + ' y_' + str(2))
    # recovery_complete_index = result['w/o']['recovery_complete_index']
    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # y_arr = [x[exp.output_index] for x in result['w/o']['outputs'][:recovery_complete_index + 1]]
    ref = [x[2] for x in exp.model.refs[:recovery_complete_index + 1]]
    plt.plot(t_arr, ref, color='black', linestyle='dashed')
    # plt.plot(t_arr, y_arr, label='w/o')

    recovery_complete_index = result['w/']['recovery_complete_index']
    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    y_arr = [x[2] for x in result['w/']['outputs'][:recovery_complete_index + 1]]
    plt.plot(t_arr, y_arr, label='w/')

    plt.vlines(exp.attack_start_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed')
    plt.vlines(exp.recovery_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted',
               linewidth=2.5)
    # plt.hlines(exp.strip[0], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    # plt.hlines(exp.strip[1], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    # plt.ylim(exp.y_lim)
    plt.xlim(exp.x_lim, recovery_complete_index * exp.dt)
    plt.legend(loc='best')
    plt.show()
    # output

