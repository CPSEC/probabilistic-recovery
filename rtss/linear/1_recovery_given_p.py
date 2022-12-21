import matplotlib.pyplot as plt
import numpy as np
from utils.formal.zonotope import Zonotope
from utils.formal.reachability import ReachableSet
from utils.formal.gaussian_distribution import GaussianDistribution
from rtss.settings import motor_speed_bias as msb
from rtss.settings import aircraft_pitch_bias as apb
from rtss.settings import boeing747_bias as boeb
from rtss.settings import heat_bias as hb
from rtss.settings import platoon_bias as pltb
from rtss.settings import rlc_circuit_bias as rcb
from rtss.settings import quadrotor_bias as qdb
from rtss.settings import quadruple_tank_bia,s as qtb
from rtss.settings import f16_bias as f16b
from rtss.settings import lane_keeping as lkp

exps = [lkp]
# exps = [msb]
# exps = [apb]
# exps = [boeb]
# exps = [hb]
# exps = [pltb]
# exps = [rcb]
# exps = [qdb]
# exps = [qtb]
# exps = [f16b]
for exp in exps:
    print('=' * 20, exp.name, '=' * 20)

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
            k, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=exp.P_given, max_k=40)
            recovery_complete_index = exp.recovery_index + k
            print('k=', k, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)
            print('recovery_complete_index=', recovery_complete_index)
            if not satisfy:
                print('>>>> P is less than P_given!')
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

    # plot
    t_arr = np.linspace(0, exp.dt*recovery_complete_index, recovery_complete_index + 1)
    # output

    ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
    y_arr = [x[exp.output_index] for x in exp.model.outputs[:recovery_complete_index + 1]]
    plt.rcParams.update({'font.size': 18})  # front sizw
    fig = plt.figure(figsize=(8, 4))
    # plt.title(exp.name+' y_'+str(exp.output_index))
    plt.ylabel(exp.y_label)
    plt.plot(t_arr, ref, color='black', linestyle='dashed')
    plt.plot(t_arr, y_arr)
    plt.vlines(exp.attack_start_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed')
    plt.vlines(exp.recovery_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted', linewidth=2.5)
    plt.hlines(exp.strip[0], exp.x_lim, recovery_complete_index*exp.dt, colors='grey')
    plt.hlines(exp.strip[1], exp.x_lim, recovery_complete_index * exp.dt, colors='grey')
    plt.ylim(exp.y_lim)
    plt.xlim(exp.x_lim, recovery_complete_index*exp.dt)
    plt.savefig('./fig/'+exp.name+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # control input
    # for i in range(exp.model.m):
    #     u_arr = [x[i] for x in exp.model.inputs[:recovery_complete_index + 1]]
    #     fig = plt.figure()
    #     plt.title(exp.name+' u_'+str(i))
    #     plt.plot(t_arr, u_arr)
    #     plt.show()

