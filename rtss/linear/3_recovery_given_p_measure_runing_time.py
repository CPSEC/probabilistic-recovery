import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["RANDOM_SEED"] = '0'   # for reproducibility


from time import perf_counter
import json
from utils.formal.zonotope import Zonotope
from utils.formal.reachability import ReachableSet
from utils.formal.gaussian_distribution import GaussianDistribution
from settings_baseline import motor_speed_bias as msb
from settings_baseline import quadruple_tank_bias as qtb
from settings_baseline import f16_bias as f16b
from settings_baseline import aircraft_pitch_bias as apb
from settings_baseline import boeing747_bias as boeb
from settings_baseline import rlc_circuit_bias as rcb
from settings_baseline import quadrotor_bias as qdb
from settings_baseline import heat_bias as htb



# exps = [msb, qtb, f16b, apb, boeb, qdb]
# exps = [f16b]
exps = [qtb]
given_Ps = [0.6, 0.8, 0.95]
result = {}
plot = False

for exp in exps:
    print('=' * 20, exp.name, '=' * 20)
    result[exp.name] = {}

    # compute offline
    u_lo = exp.model.controller.control_lo
    u_up = exp.model.controller.control_up
    U = Zonotope.from_box(u_lo, u_up)
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=exp.max_recovery_step+2)
    k = None
    rec_u = None
    recovery_complete_index = None

    for given_P in given_Ps:
        result[exp.name][given_P] = {}
        exp.model.reset()
        k = None
        rec_u = None
        recovery_complete_index = None
        print('-' * 10, 'given_P =', given_P, '-' * 10)
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.recovery_index:
                print('recovery_start_state=', exp.recovery_index)
                print('recovery_start_state=', exp.model.cur_x)
                x_0 = GaussianDistribution(exp.model.cur_x, np.zeros((exp.model.n, exp.model.n)))
                reach.init(x_0, exp.s)
                # k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=exp.P_given, max_k=40)
                tic = perf_counter()
                k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=given_P, max_k=exp.max_recovery_step)
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
                rec_u_index = i-exp.recovery_index
                u = rec_u[rec_u_index]
                exp.model.evolve(u)
                print(exp.model.cur_x)
            else:
                exp.model.evolve()

        if plot:
            # plot
            t_arr = np.linspace(0, exp.dt*recovery_complete_index, recovery_complete_index + 1)
            # output

            ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
            y_arr = [x[exp.output_index] for x in exp.model.outputs[:recovery_complete_index + 1]]
            fig = plt.figure()
            plt.title(exp.name+' y_'+str(exp.output_index))
            plt.plot(t_arr, ref, t_arr, y_arr)
            plt.show()

            # control input
            for i in range(exp.model.m):
                u_arr = [x[i] for x in exp.model.inputs[:recovery_complete_index + 1]]
                fig = plt.figure()
                plt.title('mak_k=' + str(given_P) + ' ' + exp.name + ' u_' + str(i))
                plt.plot(t_arr, u_arr)
                plt.show()

print(json.dumps(result, indent=4, sort_keys=True))
    # # plot
    # t_arr = np.linspace(0, exp.dt*recovery_complete_index, recovery_complete_index + 1)
    # # output
    #
    # ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
    # y_arr = [x[exp.output_index] for x in exp.model.outputs[:recovery_complete_index + 1]]
    # fig = plt.figure()
    # plt.title(exp.name+' y_'+str(exp.output_index))
    # plt.plot(t_arr, ref, t_arr, y_arr)
    # plt.show()
    #
    # # control input
    # for i in range(exp.model.m):
    #     u_arr = [x[i] for x in exp.model.inputs[:recovery_complete_index + 1]]
    #     fig = plt.figure()
    #     plt.title(exp.name+' u_'+str(i))
    #     plt.plot(t_arr, u_arr)
    #     plt.show()

