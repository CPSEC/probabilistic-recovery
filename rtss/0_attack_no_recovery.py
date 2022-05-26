import matplotlib.pyplot as plt
import numpy as np
from rtss.settings import motor_speed_bias as msb
np.random.seed(0)

exps = [msb]

for exp in exps:
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
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
        plt.plot(t_arr, ref, t_arr, y_arr)
        plt.show()
    # control input
    for i in range(exp.model.m):
        u_arr = [x[i] for x in exp.model.inputs[:exp.max_index + 1]]
        fig = plt.figure()
        plt.title(exp.name+' u_'+str(i))
        plt.plot(t_arr, u_arr)
        plt.show()

