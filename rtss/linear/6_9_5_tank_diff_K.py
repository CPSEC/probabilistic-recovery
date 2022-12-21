from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import csv
import time
from time import perf_counter

from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.observers.full_state_bound import Estimator
from utils.controllers.LP_cvxpy import LP
from utils.controllers.MPC_cvxpy import MPC
from utils.attack import Attack
from utils.formal.strip import Strip

from simulators.linear.quadruple_tank import QuadrupleTank

exp_num = 100

class quadruple_tank_bias:
    # needed by 0_attack_no_recovery
    max_index = 300
    dt = 1
    ref = [np.array([7, 7])] * 1001 + [np.array([7, 7])] * 1000

    control_lo = np.array([0, 0])
    control_up = np.array([2, 2])
    attack_start_index = 150
    bias = np.array([-5.0, 0])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 170

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

    def __init__(self, gamma):
        noise = {
            'process': {
                'type': 'white',
                'param': {'C': np.eye(4)*gamma}
            }
        }
        self.model = QuadrupleTank('test', self.dt, self.max_index, noise)
        self.model.controller.set_control_limit(self.control_lo, self.control_up)
        self.name = f'{gamma}'

# logger
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.FATAL)

# creat file
rst_file = '../res/tank_DIFF_K_P.csv'
overhead_file = '../res/tank_DIFF_K_overhead.csv'
headers = ['K', 'P']
overhead_headers = ['K', 'time']
with open(rst_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

with open(overhead_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(overhead_headers)

for i in range(exp_num):
    rseed = np.uint32(int(time.time()))
    np.random.seed(rseed)
    # gammas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    # exps = [quadruple_tank_bias(g) for g in gammas]
    gamma = 0.04
    exp = quadruple_tank_bias(gamma)
    baselines = ['none', 'oprp-open', 'oprp']
    # baselines = [ 'lp', 'lqr']
    baselines = ['oprp-open']
    colors = {'none': 'red', 'lp': 'cyan', 'lqr': 'blue', 'ssr': 'orange', 'oprp': 'violet', 'oprp-open': 'purple'}
    result = {}  # for print or plot
    DiFF_K = list(range(1, 16))
    for diff_k in DiFF_K:   # todo: diff K
        result[diff_k] = {}
        exp_rst = result[diff_k]

        #  =================  no_recovery  ===================
        if 'none' in baselines:
        # if True:
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

        #  =================  LP_recovery  ===================
        exp.model.reset()

        # required objects
        A = exp.model.sysd.A
        B = exp.model.sysd.B
        est = Estimator(A, B, max_k=150, epsilon=1e-7)

        # init variables
        recovery_complete_index = np.inf
        rec_u = None

        if 'lp' in baselines:
            bl = 'lp'
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

                    # State reconstruction
                    us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
                    x_0 = exp.model.states[exp.attack_start_index - 1]
                    x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                    logger.debug(f'reconstructed state={x_cur}')

                    # deadline estimate
                    safe_set_lo = exp.safe_set_lo
                    safe_set_up = exp.safe_set_up
                    control = exp.model.inputs[i - 1]
                    k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, 100)
                    recovery_complete_index = exp.recovery_index + k
                    logger.debug(f'deadline={k}')

                    # get recovery control sequence
                    lp_settings = {
                        'Ad': A, 'Bd': B,
                        'N': k,
                        'ddl': k, 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                        'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                        'control_lo': exp.control_lo, 'control_up': exp.control_up,
                        'ref': exp.recovery_ref
                    }
                    lp = LP(lp_settings)
                    _ = lp.update(feedback_value=x_cur)
                    rec_u = lp.get_full_ctrl()
                    rec_x = lp.get_last_x()
                    logger.debug(f'expected recovery state={rec_x}')

                if exp.recovery_index <= i < recovery_complete_index:
                    rec_u_index = i - exp.recovery_index
                    u = rec_u[rec_u_index]
                    exp.model.evolve(u)
                else:
                    if i == recovery_complete_index:
                        logger.debug(f'state after recovery={exp.model.cur_x}')
                        step = recovery_complete_index - exp.recovery_index
                        logger.debug(f'use {step} steps to recover.')
                    exp.model.evolve()

            exp_rst[bl] = {}
            exp_rst[bl]['states'] = deepcopy(exp.model.states)
            exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
            exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
            exp_rst[bl]['time'] = {}
            exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index

        #  =================  LQR_recovery  ===================
        # did not add maintainable time estimation, let it to be 3
        maintain_time = 3
        exp.model.reset()

        # init variables
        recovery_complete_index = np.inf
        rec_u = None

        if 'lqr' in baselines:
            bl = 'lqr'
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

                    # State reconstruction
                    us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
                    x_0 = exp.model.states[exp.attack_start_index - 1]
                    x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                    logger.debug(f'reconstructed state={x_cur}')

                    # deadline estimate
                    safe_set_lo = exp.safe_set_lo
                    safe_set_up = exp.safe_set_up
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
                    _ = lqr.update(feedback_value=x_cur)
                    rec_u = lqr.get_full_ctrl()
                    rec_x = lqr.get_last_x()
                    logger.debug(f'expected recovery state={rec_x}')

                if i == recovery_complete_index:
                    logger.debug(f'state after recovery={exp.model.cur_x}')
                    step = recovery_complete_index - exp.recovery_index
                    logger.debug(f'use {step} steps to recover.')

                if exp.recovery_index <= i < recovery_complete_index + maintain_time:
                    rec_u_index = i - exp.recovery_index
                    u = rec_u[rec_u_index]
                    exp.model.evolve(u)
                else:
                    exp.model.evolve()

            exp_rst[bl] = {}
            exp_rst[bl]['states'] = deepcopy(exp.model.states)
            exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
            exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
            exp_rst[bl]['time'] = {}
            exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index + maintain_time

        #  =================  Software_sensor_recovery  ===================
        exp.model.reset()

        # required objects
        def in_target_set(target_lo, target_hi, x_cur):
            res = True
            for i in range(len(x_cur)):
                if target_lo[i] > x_cur[i] or target_hi[i] < x_cur[i]:
                    res = False
                    break
            return res

        # init variables
        recovery_complete_index = np.inf
        last_predicted_state = None

        if 'ssr' in baselines:
            bl = 'ssr'
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

                    # State reconstruction
                    us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index-1]
                    x_0 = exp.model.states[exp.attack_start_index - 1]
                    x_cur = est.estimate_wo_bound(x_0, us)
                    logger.debug(f'one before reconstructed state={x_cur}')
                    last_predicted_state = deepcopy(x_cur)

                if exp.recovery_index <= i <= recovery_complete_index:
                    # check if it is in target set
                    # if in_target_set(exp.target_set_lo, exp.target_set_up, last_predicted_state):
                    #     recovery_complete_index = i
                    #     logger.debug('state after recovery={exp.model.cur_x}')
                    #     step = recovery_complete_index - exp.recovery_index
                    #     logger.debug(f'use {step} steps to recover.')
                    us = [exp.model.inputs[i - 1]]
                    x_0 = last_predicted_state
                    x_cur = est.estimate_wo_bound(x_0, us)
                    exp.model.cur_feedback = exp.model.sysd.C @ x_cur
                    last_predicted_state = deepcopy(x_cur)
                    # print(f'{exp.model.cur_u}')
                exp.model.evolve()

            exp_rst[bl] = {}
            exp_rst[bl]['states'] = deepcopy(exp.model.states)
            exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
            exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
            exp_rst[bl]['time'] = {}
            exp_rst[bl]['time']['recovery_complete'] = exp.max_index-1
            # print(f'{recovery_complete_index}')

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
        max_P = 0

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
                    reach.init(x_cur_update, exp.s)
                    k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.max_recovery_step)
                    # print(f"{k=}, {z_star=}, {P=}")
                    max_P = P
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
            exp_rst[bl]['P'] = max_P


        #  =================  Optimal_probabilistic_recovery - OPEN LOOP ===================
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
        reach = ReachableSet(A, B, U, W, max_step=100)

        # init variables
        recovery_complete_index = np.inf
        x_cur_update = None
        exp_rst[diff_k] = {}

        if 'oprp-open' in baselines:
            bl = 'oprp-open'
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

                    # state reconstruction
                    b_recovery_t = perf_counter()
                    us = exp.model.inputs[exp.attack_start_index-1:exp.recovery_index]
                    ys = (kf_C @ exp.model.states[exp.attack_start_index:exp.recovery_index + 1].T).T
                    x_0 = exp.model.states[exp.attack_start_index-1]
                    x_res, P_res = kf.multi_steps(x_0, np.zeros_like(A), us, ys)
                    x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
                    logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")


                    reach.init(x_cur_update, exp.s)
                    k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=diff_k)   # todo: K
                    rec_u = U.alpha_to_control(alpha)
                    e_recovery_t = perf_counter()
                    recovery_t = e_recovery_t - b_recovery_t
                    recovery_complete_index = i + k
                    max_P = P

                if exp.recovery_index <= i < recovery_complete_index:
                    rec_u_index = i - exp.recovery_index
                    u = rec_u[rec_u_index]
                    exp.model.evolve(u)
                else:
                    exp.model.evolve()


            # exp_rst[bl]['states'] = deepcopy(exp.model.states)
            # exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
            # exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
            # exp_rst[bl]['time'] = {}
            # exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index
            exp_rst[diff_k]['overhead'] = recovery_t
            exp_rst[diff_k]['P'] = max_P


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

    # baselines = ['oprp-open', 'oprp']
    baselines = ['oprp-open']
    blables = {'oprp-open': 'Open loop', 'oprp': 'Close loop'}
    one_rst = []
    overhead_rst = []
    for diff_k in DiFF_K:     # TODO： K
        for bl in baselines:
            one_rst.append([diff_k,  result[diff_k][diff_k]['P']])
            overhead_rst.append([diff_k, result[diff_k][diff_k]['overhead']*1000])

    with open(rst_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(one_rst)


    with open(overhead_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(overhead_rst)
    # rst = np.array(one_rst)
    # # print(rst[:, 0])
    # plt.plot(rst[:, 0], rst[:, 1])
    # plt.show()