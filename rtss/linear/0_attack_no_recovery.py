import matplotlib.pyplot as plt
import numpy as np
from rtss.settings_baseline import quadruple_tank_bias as qtb

# exps = [msb, qtb, f16b]
# exps = [msb]
exps = [qtb]
# exps = [f16b]
# exps = [apb]
# exps = [htb]
for exp in exps:
    print('='*20, exp.name, '='*20)
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        if i == exp.attack_start_index-1:
            print('normal_state=', exp.model.cur_x)
        if i == exp.recovery_index:
            print('recovery_start_state=', exp.recovery_index)
            print('recovery_start_state=', exp.model.cur_x)
        exp.model.evolve()

    # plot
    t_arr = np.linspace(0, exp.dt*exp.max_index, exp.max_index + 1)
    # output
    for i in range(exp.model.p):
        ref = [x[i] for x in exp.model.refs[:exp.max_index + 1]]
        y_arr = [x[i] for x in exp.model.outputs[:exp.max_index + 1]]
        fig = plt.figure()
        plt.title(exp.name+' y_'+str(i))
        plt.plot(t_arr, ref, color='black', linestyle='dashed')
        plt.plot(t_arr, y_arr)
        plt.show()
    # control input
    for i in range(exp.model.m):
        u_arr = [x[i] for x in exp.model.inputs[:exp.max_index + 1]]
        fig = plt.figure()
        plt.title(exp.name+' u_'+str(i))
        plt.plot(t_arr, u_arr)
        plt.show()

