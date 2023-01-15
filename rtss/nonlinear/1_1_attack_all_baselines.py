import os, sys
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from utils.observers import full_state_nonlinear_from_gaussian as fsn
from utils.observers import full_state_bound as fsb
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.zonotope import Zonotope
from utils.formal.reachability import ReachableSet
from utils.observers.extended_kalman_filter import ExtendedKalmanFilter
from utils.controllers.MPC_cvxpy import MPC

# logger
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# random seed
os.environ["RANDOM_SEED"] = '0'  # for reproducibility

# import benchmarks from settings
from settings import cstr

# simulation settings
baselines = ['none', 'oprp_ol', 'oprp_cl', 'lqr', 'vsr']
# baselines = ['vsr']
exps = [cstr]
colors = {'none': 'red', 'oprp_cl': 'blue', 'oprp_ol': 'cyan', 'lqr': 'green', 'vsr': 'orange'}  # 'lp': 'cyan', 'lqr': 'green', 'ssr': 'orange', 'mpc': 'blue'}
result = {}  # for plotting figures

for exp in exps:
    result[exp.name] = {}
    exp_rst = result[exp.name]

    # ---------  attack + no recovery  -------------
    if True:
        bl = 'none'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
            exp.model.evolve()
        exp_rst[bl] = {}
        exp_rst[bl]['refs'] = deepcopy(exp.model.refs)
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = exp.max_index - 1

    # ---------  attack + OPRP recovery (open loop)  -------------
    if 'oprp_ol' in baselines:
        bl = 'oprp_ol'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")

        # init some variables
        recovery_complete_index = exp.max_index - 1
        x_cur = None
        sysd = None
        u_lo = exp.model.controller.control_lo
        u_up = exp.model.controller.control_up
        U = Zonotope.from_box(u_lo, u_up)
        W = exp.model.p_noise_dist

        # init for recovery
        exp.model.reset()
        non_est = fsn.Estimator(exp.model.ode, exp.model.n, exp.model.m, exp.dt, W)

        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')

            # state reconstruction & linearization
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
                us = exp.model.inputs[exp.attack_start_index - 1:i]
                x_0 = GaussianDistribution(exp.model.states[exp.attack_start_index - 1], np.zeros((exp.model.n, exp.model.n)))
                x_cur, sysd = non_est.estimate(x_0, us)
                logger.debug(f'recovered_cur_state={x_cur}')
            if exp.recovery_index < i < recovery_complete_index:
                u_last = [exp.model.inputs[i - 1]]
                x_cur, sysd = non_est.estimate(x_cur, u_last)

            # call OPRP (did not consider good sensor)
            if exp.recovery_index <= i < recovery_complete_index:
                reach = ReachableSet(sysd.A, sysd.B, U, W, max_step=100, c=sysd.c)
                reach.init(x_cur, exp.s)
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
                recovery_complete_index = i + k
                rec_u = U.alpha_to_control(alpha)
                u = rec_u[0]
                exp.model.evolve(u)
                logger.debug(f'recovering {i=},{u=},{x_cur.miu=},x_grnd={exp.model.states[i]}')
            else:
                if i == recovery_complete_index:
                    logger.debug(f'state after recovery: {exp.model.cur_x}')
                    step = recovery_complete_index - exp.recovery_index
                    logger.debug(f'use {step} steps to recover.')
                exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['refs'] = deepcopy(exp.model.refs)
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index

    # ---------  attack + OPRP recovery (close loop) -------------
    if 'oprp_cl' in baselines:
        bl = 'oprp_cl'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")

        # init some variables
        recovery_complete_index = exp.max_index - 1
        x_cur = None
        sysd = None
        u_lo = exp.model.controller.control_lo
        u_up = exp.model.controller.control_up
        U = Zonotope.from_box(u_lo, u_up)
        W = exp.model.p_noise_dist

        # init for recovery
        exp.model.reset()

        jh = exp.jh
        C = jh(0, 0)
        Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros((exp.nx, exp.nx))
        R = np.zeros((C.shape[0], C.shape[0]))
        kalman_filter = ExtendedKalmanFilter(exp.model.f, exp.model.jfx, jh, Q, R)

        non_est = fsn.Estimator(exp.model.ode, exp.model.n, exp.model.m, exp.dt, W, kf=kalman_filter, jfx=exp.model.jfx,
                                jfu=exp.model.jfu)

        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')

            # state reconstruction & linearization
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
                us = exp.model.inputs[exp.attack_start_index - 1:i]
                ys = (C @ exp.model.states[exp.attack_start_index:exp.recovery_index + 1].T).T
                x_0 = GaussianDistribution(exp.model.states[exp.attack_start_index - 1],
                                           np.zeros((exp.model.n, exp.model.n)))
                x_cur, sysd = non_est.estimate(x_0, us, ys)
                logger.debug(f'recovered_cur_state={x_cur}')
            if exp.recovery_index < i < recovery_complete_index:
                u_last = [exp.model.inputs[i - 1]]
                y_last = [C @ exp.model.cur_x]
                x_cur, sysd = non_est.estimate(x_cur, u_last, y_last)

            # call OPRP (did not consider good sensor)
            if exp.recovery_index <= i < recovery_complete_index:
                reach = ReachableSet(sysd.A, sysd.B, U, W, max_step=100, c=sysd.c)
                reach.init(x_cur, exp.s)
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
                recovery_complete_index = i + k
                rec_u = U.alpha_to_control(alpha)
                u = rec_u[0]
                exp.model.evolve(u)
                logger.debug(f'recovering {i=},{u=},{x_cur.miu=}')
            else:
                if i == recovery_complete_index:
                    logger.debug(f'state after recovery: {exp.model.cur_x}')
                    step = recovery_complete_index - exp.recovery_index
                    logger.debug(f'use {step} steps to recover.')
                exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['refs'] = deepcopy(exp.model.refs)
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index

    # ---------  attack + RTR-LQR recovery  -------------
    if 'lqr' in baselines:
        bl = 'lqr'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")

        # init some variables
        recovery_complete_index = exp.max_index - 1
        est = None
        maintain_time = 3

        # init for recovery
        exp.model.reset()
        jh = exp.jh
        C = jh(0, 0)

        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
                exp.model.linearize_at(exp.model.cur_x, exp.model.cur_u)
                A = exp.model.sysd.A
                B = exp.model.sysd.B
                c = exp.model.sysd.c
                est = fsb.Estimator(A, B, max_k=150, epsilon=1e-7, c=c)

            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
                # state reconstruction
                us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
                x_0 = exp.model.states[exp.attack_start_index - 1]
                x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                logger.debug(f'reconstructed state={x_cur}')

                # deadline calculation
                safe_set_lo = exp.safe_set_lo
                safe_set_up = exp.safe_set_up
                control = exp.model.inputs[i - 1]
                exp.model.linearize_at(x_cur, exp.model.cur_u)
                A = exp.model.sysd.A
                B = exp.model.sysd.B
                c = exp.model.sysd.c
                est = fsb.Estimator(A, B, max_k=150, epsilon=1e-7, c=c)
                k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, max_k=100)
                recovery_complete_index = exp.recovery_index + k
                logger.debug(f'deadline={k}')

                # get recovery control sequence
                Q_lqr = exp.Q #np.diag([1]*exp.nx)
                QN_lqr = exp.QN  #np.diag([1]*exp.nx)
                R_lqr = exp.R #  np.diag([1]*exp.nu)
                lqr_settings = {
                    'Ad': A, 'Bd': B, 'c_nonlinear': c,
                    'Q': Q_lqr, 'QN': QN_lqr, 'R': R_lqr,
                    # 'Q': exp.Q, 'QN': exp.QN, 'R': exp.R,
                    'N': k + maintain_time,
                    'ddl': k, 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                    'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                    'control_lo': exp.control_lo, 'control_up': exp.control_up,
                    'ref': exp.recovery_ref
                }
                lqr = MPC(lqr_settings)
                _ = lqr.update(feedback_value=x_cur)
                rec_u_lqr = lqr.get_full_ctrl()
                rec_x = lqr.get_last_x()
                logger.debug(f'expected recovery state={rec_x}')

            if i == recovery_complete_index:
                logger.debug(f'state after recovery={exp.model.cur_x}')
                step = recovery_complete_index - exp.recovery_index
                logger.debug(f'use {step} steps to recover.')

            if exp.recovery_index <= i < recovery_complete_index + maintain_time:
                rec_u_index = i - exp.recovery_index
                u = rec_u_lqr[rec_u_index]
                exp.model.evolve(u)
            else:
                exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index + maintain_time


    # ---------  attack + virtual sensor recovery  -------------
    if 'vsr' in baselines:
        bl = 'vsr'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")

        # required objects
        def in_target_set(target_lo, target_hi, x_cur):
            res = True
            for i in range(len(x_cur)):
                if target_lo[i] > x_cur[i] or target_hi[i] < x_cur[i]:
                    res = False
                    break
            return res

        # init for recovery
        exp.model.reset()
        recovery_complete_index = exp.max_index - 1
        last_predicted_state = None

        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
                exp.model.linearize_at(exp.model.cur_x, exp.model.cur_u)
                A = exp.model.sysd.A
                B = exp.model.sysd.B
                c = exp.model.sysd.c
                est = fsb.Estimator(A, B, max_k=150, epsilon=1e-7, c=c)

            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
                # state reconstruction
                us = exp.model.inputs[exp.attack_start_index - 1:i]
                xs = exp.model.states[exp.attack_start_index - 1:i+1]
                x_0 = exp.model.states[exp.attack_start_index - 1]
                x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                last_predicted_state = deepcopy(x_cur)

            if exp.recovery_index <= i <= recovery_complete_index:
                # check if it is in target set
                if in_target_set(exp.target_set_lo, exp.target_set_up, last_predicted_state):
                    recovery_complete_index = i
                    logger.debug('state after recovery={exp.model.cur_x}')
                    step = recovery_complete_index - exp.recovery_index
                    logger.debug(f'use {step} steps to recover.')

                # update linear model
                if exp.recovery_index == i:
                    exp.model.linearize_at(last_predicted_state, exp.model.cur_u)
                    A = exp.model.sysd.A
                    B = exp.model.sysd.B
                    c = exp.model.sysd.c
                    est = fsb.Estimator(A, B, max_k=150, epsilon=1e-7, c=c)

                us = np.array([exp.model.inputs[i - 1]])
                xs = exp.model.states[i - 1:i + 1]
                x_0 = last_predicted_state
                x_cur = est.estimate_wo_bound(x_0, us)
                exp.model.cur_feedback = x_cur  # exp.model.sysd.C @ x_cur
                last_predicted_state = deepcopy(x_cur)
            exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index


    # ==================== plot =============================
    plt.rcParams.update({'font.size': 24})  # front size
    fig = plt.figure(figsize=(10, 5))

    # plot reference
    t_arr = np.linspace(0, exp.dt * exp.max_index, exp.max_index + 1)[:exp.max_index]
    ref = [x[exp.ref_index] for x in exp_rst['none']['refs'][:exp.max_index]]
    plt.plot(t_arr, ref, color='grey', linestyle='dashed')
    # plot common part (before recovery)
    t_arr_common = t_arr[:exp.recovery_index + 1]
    output = [x[exp.output_index] for x in exp_rst['none']['outputs'][:exp.recovery_index + 1]]
    plt.plot(t_arr_common, output, color='black')
    # plot attack / recovery
    if exp.y_lim:
        plt.vlines(exp.attack_start_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed',
                   linewidth=2)
        plt.vlines(exp.recovery_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted',
                   linewidth=2)

        # recovery_complete_index = exp.recovery_index + deadline_for_all_methods
        # # print(recovery_complete_index)
        # plt.vlines((recovery_complete_index) * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='blue', linestyle='dotted',
        #            linewidth=2)
        # plt.vlines((recovery_complete_index + maintain_time) * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='black',
        #            linestyle='dotted', linewidth=2)

    # strip
    cnt = len(t_arr)
    y1 = [exp.strip[0]] * cnt
    y2 = [exp.strip[1]] * cnt
    plt.fill_between(t_arr, y1, y2, facecolor='green', alpha=0.1)

    for bl in baselines:
        end_time = exp_rst[bl]['time']['recovery_complete']
        t_arr_tmp = t_arr[exp.recovery_index:end_time + 1]
        output = [x[exp.output_index] for x in exp_rst[bl]['outputs'][exp.recovery_index:end_time + 1]]
        # output = [x[exp.output_index] for x in exp_rst[bl]['states'][exp.recovery_index:end_time + 1]]
        plt.plot(t_arr_tmp, output, color=colors[bl], label=bl)

    if exp.y_lim:
        plt.ylim(exp.y_lim)
    if exp.x_lim:
        plt.xlim(exp.x_lim)
    #     # updated_x_lim = (exp.x_lim[0], exp.dt * (recovery_complete_index + maintain_time))
    #     updated_x_lim = (exp.x_lim[0], exp.dt * (exp.max_index-1))
    #     plt.xlim(updated_x_lim)

    # plt.legend()
    plt.ylabel(exp.y_label)
    plt.xlabel('Time [sec]', loc='right', labelpad=-55)
    plt.legend()

    plt.savefig(f'../fig/nonlinear/{exp.name}_all.png', format='png', bbox_inches='tight')
    plt.show()


    # save state
    # res/data
    path = os.path.join('../res/data', exp.name)
    if not os.path.exists(path):
        os.makedirs(path)

    import csv
    for bl in baselines:
        file_name = os.path.join(path, f'states_{bl}.csv')
        end_time = exp_rst[bl]['time']['recovery_complete']
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['k', 'time']+exp.state_names)
            for i in range(end_time + 1):
                states = list(exp_rst[bl]['states'][i])
                writer.writerow([i, i*exp.dt]+states)



