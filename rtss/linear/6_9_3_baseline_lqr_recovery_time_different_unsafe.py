from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping, f16_bias, aircraft_pitch_bias, boeing747_bias, platoon_bias
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.observers.full_state_bound import Estimator
from utils.controllers.LP_cvxpy import LP
from utils.controllers.MPC_cvxpy import MPC

exps = [quadruple_tank_bias]
# baselines = ['none', 'lp', 'lqr', 'ssr', 'oprp', 'fprp']
baselines = ['lqr', 'oprp']
# baselines = [ 'lp', 'lqr']
colors = {'none': 'red', 'lp': 'cyan', 'lqr': 'blue', 'ssr': 'orange', 'oprp': 'purple', 'fprp': 'violet'}
result = {}  # for print or plot
max_l0s = [15.8, 16, 17, 18, 19, 20, 21, 22]    # only for quadruple_tank_bias

# logger
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for exp in exps:
    result[exp.name] = {}
    exp_rst = result[exp.name]

    in_strip = False
    first_in_strip_index = None

    #  =================  no_recovery  ===================
    # if 'none' in baselines:
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
        exp_rst[bl]['time']['recovery_complete'] = exp.max_index-1

    # required objects
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    est = Estimator(A, B, max_k=150, epsilon=1e-7)

    #  =================  LQR_recovery  ===================
    # did not add maintainable time estimation, let it to be 3

    if 'lqr' in baselines:
        bl = 'lqr'
        exp_rst[bl] = {}
        for max_l0 in max_l0s:
            maintain_time = 3
            exp.model.reset()

            # init variables
            recovery_complete_index = np.inf
            rec_u = None
            in_strip = False
            k = -1
            solved = True

            exp_name = f" {bl} {max_l0=} {exp.name} "
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

                    # State reconstruction
                    us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
                    x_0 = exp.model.states[exp.attack_start_index - 1]
                    x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                    logger.debug(f'reconstructed state={x_cur}')

                    # deadline estimate
                    safe_set_lo = exp.safe_set_lo
                    safe_set_up = exp.safe_set_up
                    safe_set_up[0] = max_l0    #   change safe set !!!!
                    control = exp.model.inputs[i - 1]
                    k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, 100)
                    recovery_complete_index = exp.recovery_index + k
                    logger.debug(f'deadline={k}')
                    # maintainable time compute

                    # get recovery control sequence
                    lqr_settings = {
                        'Ad': A, 'Bd': B,
                        'Q': exp.Q, 'QN': exp.QN, 'R': exp.R,
                        'N': k + 3,
                        'ddl': k, 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                        'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                        'control_lo': exp.control_lo, 'control_up': exp.control_up,
                        'ref': exp.recovery_ref
                    }
                    lqr = MPC(lqr_settings)
                    try:
                        _ = lqr.update(feedback_value=x_cur)
                        rec_u = lqr.get_full_ctrl()
                        rec_x = lqr.get_last_x()
                        logger.debug(f'expected recovery state={rec_x}')
                    except:
                        solved = False
                        break

                if i == recovery_complete_index:
                    logger.debug(f'state after recovery={exp.model.cur_x}')
                    step = recovery_complete_index - exp.recovery_index
                    logger.debug(f'use {step} steps to recover.')

                if exp.recovery_index <= i < recovery_complete_index + maintain_time:
                    if not in_strip and exp.s.in_strip(exp.model.cur_x):
                        in_strip = True
                        first_in_strip_index = i
                    rec_u_index = i - exp.recovery_index
                    u = rec_u[rec_u_index]
                    exp.model.evolve(u)
                else:
                    exp.model.evolve()


            exp_rst[bl][max_l0] = {}
            exp_rst[bl][max_l0]['states'] = deepcopy(exp.model.states)
            exp_rst[bl][max_l0]['outputs'] = deepcopy(exp.model.outputs)
            exp_rst[bl][max_l0]['inputs'] = deepcopy(exp.model.inputs)
            exp_rst[bl][max_l0]['solved'] = solved
            exp_rst[bl][max_l0]['time'] = {}
            exp_rst[bl][max_l0]['time']['k'] = k
            if solved:
                exp_rst[bl][max_l0]['time']['recovery_step'] = first_in_strip_index - exp.recovery_index if in_strip else 'N/A'
            else:
                exp_rst[bl][max_l0]['time']['recovery_step'] = -1


    #  =================  Optimal_probabilistic_recovery  ===================
    exp.model.reset()

    # required objects
    kf_C = exp.kf_C
    C = exp.model.sysd.C
    D = exp.model.sysd.D
    kf_Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
    kf_R = exp.kf_R
    kf = KalmanFilter(A, B, kf_C, D, kf_Q, kf_R)
    U = Zonotope.from_box(exp.control_lo, exp.control_up)
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=exp.max_recovery_step + 2)

    # init variables
    recovery_complete_index = np.inf
    x_cur_update = None
    in_strip = False

    if 'oprp' in baselines:
        bl = 'oprp'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')

            # state reconstruct
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')

                us = exp.model.inputs[exp.attack_start_index-1:exp.recovery_index]
                ys = (kf_C @ exp.model.states[exp.attack_start_index:exp.recovery_index + 1].T).T
                x_0 = exp.model.states[exp.attack_start_index-1]
                x_res, P_res = kf.multi_steps(x_0, np.zeros_like(A), us, ys)
                x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
                logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")
                # x_cur_update = GaussianDistribution(exp.model.cur_x, P_res[-1])

            if exp.recovery_index < i < recovery_complete_index:
                x_cur_predict = GaussianDistribution(*kf.predict(x_cur_update.miu, x_cur_update.sigma, exp.model.cur_u))
                y = kf_C @ exp.model.cur_x
                x_cur_update = GaussianDistribution(*kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
                logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")

            if i == recovery_complete_index:
                logger.debug(f'state after recovery={exp.model.cur_x}')

            if exp.recovery_index <= i < recovery_complete_index:
                if not in_strip and exp.s.in_strip(exp.model.cur_x):
                    in_strip = True
                    first_in_strip_index = i

                reach.init(x_cur_update, exp.s)
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.max_recovery_step)
                # print(f"{k=}, {z_star=}, {P=}")
                recovery_control_sequence = U.alpha_to_control(alpha)
                recovery_complete_index = i+k

                exp.model.evolve(recovery_control_sequence[0])
                # print(f"{i=}, {recovery_control_sequence[0]=}")
            else:
                exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index
        exp_rst[bl]['time']['recovery_step'] = first_in_strip_index - exp.recovery_index if in_strip else 'N/A'


    # # ==================== plot =============================
    # plt.rcParams.update({'font.size': 24})  # front size
    # fig = plt.figure(figsize=(8, 4))
    #
    # # plot reference
    # t_arr = np.linspace(0, exp.dt * exp.max_index, exp.max_index + 1)[:exp.max_index]
    # ref = [x[exp.ref_index] for x in exp_rst['none']['refs'][:exp.max_index]]
    # plt.plot(t_arr, ref, color='grey', linestyle='dashed')
    # # plot common part (before recovery)
    # t_arr_common = t_arr[:exp.recovery_index + 1]
    # output = [x[exp.output_index] for x in exp_rst['none']['outputs'][:exp.recovery_index + 1]]
    # plt.plot(t_arr_common, output, color='black')
    # # plot attack / recovery
    # if exp.y_lim:
    #     plt.vlines(exp.attack_start_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed', linewidth=2)
    #     plt.vlines(exp.recovery_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted', linewidth=2)
    # # strip
    # cnt = len(t_arr)
    # y1 = [exp.strip[0]]*cnt
    # y2 = [exp.strip[1]]*cnt
    # plt.fill_between(t_arr, y1, y2, facecolor='green', alpha=0.1)
    #
    # for bl in baselines:
    #     end_time = exp_rst[bl]['time']['recovery_complete']
    #     t_arr_tmp = t_arr[exp.recovery_index:end_time+1]
    #     output = [x[exp.output_index] for x in exp_rst[bl]['outputs'][exp.recovery_index:end_time+1]]
    #     plt.plot(t_arr_tmp, output, color=colors[bl], label=bl)
    #
    # if exp.y_lim:
    #     plt.ylim(exp.y_lim)
    # if exp.x_lim:
    #     plt.xlim(exp.x_lim)
    #
    # # plt.legend()
    # plt.ylabel(exp.y_label)
    # plt.savefig(f'fig/baselines/{exp.name}.svg', format='svg', bbox_inches='tight')
    # plt.show()

# save result to csv
import csv
headers = ['max_l0', 'k', 'recovery_step']
rows = []
for exp_rst_name in result:
    exp_rst = result[exp_rst_name]
    for bl in baselines:
        if bl == 'lqr':
            for max_l0 in max_l0s:
                k = exp_rst[bl][max_l0]['time']['k']
                recover_step = exp_rst[bl][max_l0]['time']['recovery_step']
                row = [max_l0, k, recover_step]
                rows.append(row)
with open('../res/baseline_lqr_recovery_step_diff_unsafe.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)