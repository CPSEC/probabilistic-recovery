from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.observers.full_state_bound import Estimator
from utils.controllers.LP_cvxpy import LP
from utils.controllers.MPC_cvxpy import MPC

exps = [quadruple_tank_bias]
# baselines = ['none', 'lp', 'mpc', 'ssr', 'oprp', 'fprp']
baselines = ['none', 'lp', 'mpc']
colors = {'none': 'red', 'lp': 'cyan', 'mpc': 'blue', 'ssr': 'yellow', 'oprp': 'purple', 'fprp': 'violet'}
result = {}  # for print or plot

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

    #  =================  no_recovery  ===================
    # if 'none' in baselines:
    if True:
        exp_name = f" none + {exp.name} "
        logger.info(f"{exp_name:^40}")
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
        exp_rst['none'] = {}
        exp_rst['none']['refs'] = deepcopy(exp.model.refs)
        exp_rst['none']['states'] = deepcopy(exp.model.states)
        exp_rst['none']['outputs'] = deepcopy(exp.model.outputs)
        exp_rst['none']['inputs'] = deepcopy(exp.model.inputs)
        exp_rst['none']['time'] = {}
        exp_rst['none']['time']['recovery_complete'] = exp.max_index

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
        exp_name = f" lp + {exp.name} "
        logger.info(f"{exp_name:^40}")
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
                recovery_complete_index = exp.attack_start_index + k
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

            exp_rst['lp'] = {}
            exp_rst['lp']['states'] = deepcopy(exp.model.states)
            exp_rst['lp']['outputs'] = deepcopy(exp.model.outputs)
            exp_rst['lp']['inputs'] = deepcopy(exp.model.inputs)
            exp_rst['lp']['time'] = {}
            exp_rst['lp']['time']['recovery_complete'] = recovery_complete_index

    #  =================  MPC_recovery  ===================
    # did not add maintainable time estimation, let it to be 3
    maintain_time = 3
    exp.model.reset()

    # init variables
    recovery_complete_index = np.inf
    rec_u = None

    if 'mpc' in baselines:
        exp_name = f" mpc + {exp.name} "
        logger.info(f"{exp_name:^40}")
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
                recovery_complete_index = exp.attack_start_index + k
                logger.debug(f'deadline={k}')
                # maintainable time compute


                # get recovery control sequence
                mpc_settings = {
                    'Ad': A, 'Bd': B,
                    'Q': exp.Q, 'QN': exp.QN, 'R': exp.R,
                    'N': k + 3,
                    'ddl': k, 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                    'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                    'control_lo': exp.control_lo, 'control_up': exp.control_up,
                    'ref': np.array([14, 14, 2, 2.5])
                }
                mpc = MPC(mpc_settings)
                _ = mpc.update(feedback_value=x_cur)
                rec_u = mpc.get_full_ctrl()
                rec_x = mpc.get_last_x()
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

            exp_rst['mpc'] = {}
            exp_rst['mpc']['states'] = deepcopy(exp.model.states)
            exp_rst['mpc']['outputs'] = deepcopy(exp.model.outputs)
            exp_rst['mpc']['inputs'] = deepcopy(exp.model.inputs)
            exp_rst['mpc']['time'] = {}
            exp_rst['mpc']['time']['recovery_complete'] = recovery_complete_index + maintain_time

    #  =================  Software_sensor_recovery  ===================
    exp.model.reset()

    # init variables
    recovery_complete_index = np.inf




    # ==================== plot =============================
    plt.rcParams.update({'font.size': 18})  # front size
    fig = plt.figure(figsize=(8, 4))

    # plot reference
    t_arr = np.linspace(0, exp.dt * exp.max_index, exp.max_index + 1)[:exp.max_index]
    ref = [x[exp.ref_index] for x in exp_rst['none']['refs'][:exp.max_index]]
    plt.plot(t_arr, ref, color='grey', linestyle='dashed')
    # plot common part (before recovery)
    t_arr_common = t_arr[:exp.recovery_index + 1]
    output = [x[exp.output_index] for x in exp_rst['none']['outputs'][:exp.recovery_index + 1]]
    plt.plot(t_arr_common, output, color='black')
    # plot attack / recovery

    for bl in baselines:
        end_time = exp_rst[bl]['time']['recovery_complete']
        t_arr_tmp = t_arr[exp.recovery_index:end_time]
        output = [x[exp.output_index] for x in exp_rst[bl]['outputs'][exp.recovery_index:end_time]]
        plt.plot(t_arr_tmp, output, color=colors[bl])

    plt.show()
