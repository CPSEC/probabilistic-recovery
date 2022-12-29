import os, sys
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from utils.observers import full_state_nonlinear as fsn

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
baselines = ['none', 'oprp']
exps = [cstr]
colors = {'none': 'red', 'oprp': 'blue'}  # 'lp': 'cyan', 'lqr': 'green', 'ssr': 'orange', 'mpc': 'blue'}
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

    # ---------  attack + OPRP recovery  -------------
    if 'oprp' in baselines:
        bl = 'oprp'
        exp_name = f" {bl} {exp.name} "
        logger.info(f"{exp_name:=^40}")

        # init for recovery
        exp.model.reset()
        non_est = fsn.Estimator(exp.model.ode, exp.dt)

        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
                # state reconstruction
                us = exp.model.inputs[exp.attack_start_index - 1:i]
                x_0 = exp.model.states[exp.attack_start_index - 1]
                xs = non_est.estimate(x_0, us)
                # g_xs = exp.model.states[exp.attack_start_index:i+1] # ground truth
                x_cur = xs[-1]
                logger.debug(f'recovered_cur_state={x_cur}')


            exp.model.evolve()

        sys.exit()

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
    # # strip
    # cnt = len(t_arr)
    # y1 = [exp.strip[0]] * cnt
    # y2 = [exp.strip[1]] * cnt
    # plt.fill_between(t_arr, y1, y2, facecolor='green', alpha=0.1)

    for bl in baselines:
        end_time = exp_rst[bl]['time']['recovery_complete']
        t_arr_tmp = t_arr[exp.recovery_index:end_time + 1]
        output = [x[exp.output_index] for x in exp_rst[bl]['outputs'][exp.recovery_index:end_time + 1]]
        # output = [x[exp.output_index] for x in exp_rst[bl]['states'][exp.recovery_index:end_time + 1]]
        plt.plot(t_arr_tmp, output, color=colors[bl], label=bl)

    # if exp.y_lim:
    #     plt.ylim(exp.y_lim)
    # if exp.x_lim:
    #     # updated_x_lim = (exp.x_lim[0], exp.dt * (recovery_complete_index + maintain_time))
    #     updated_x_lim = (exp.x_lim[0], exp.dt * (exp.max_index-1))
    #     plt.xlim(updated_x_lim)

    # plt.legend()
    plt.ylabel(exp.y_label)
    plt.xlabel('Time [sec]', loc='right', labelpad=-55)
    plt.legend()

    plt.savefig(f'../fig/nonlinear/{exp.name}_all.png', format='png', bbox_inches='tight')
    plt.show()