import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
from copy import deepcopy

from settings import cstr
from settings import svl
# exps = [cstr]
exps = [svl]

labels = ['OPR-OL', 'RTR-LQR', 'VS']
markers = ["o", 'D', '^']
colors = ["C0", "C1", "C2"]
strategies = ['oprp_ol', 'lqr', 'vsr']
data_path = '../res/data'
fig_path = '../res/figs'

def dist_to_strip(x, s):
    return np.abs(s.l @ x - (s.a+s.b)/2) / np.linalg.norm(s.l)

for exp in exps:
    plt.rcParams.update({'font.size': 15})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    path = os.path.join('../res/data', exp.name)
    final_states_file_name = os.path.join(path, f'final_states.csv')
    data = pd.read_csv(final_states_file_name)
    success = np.zeros(len(strategies))
    rec_times = []
    dist_ref = []
    for s, strategy in enumerate(strategies):
        data_strat = data[data['name'] == strategy]
        success_cnt = len(data_strat[data_strat['Success'] == 1].index)
        success[s] = success_cnt / len(data_strat.index)
        print([np.array(row.iloc[5:5 + exp.nx]) for index, row in data_strat.iterrows()])
        dist = [dist_to_strip(np.array(row.iloc[5:5+exp.nx]), exp.s) for index, row in data_strat.iterrows()]
        dist_ref.append(dist)
        rec_time = np.array(data_strat['steps_recovery'])
        rec_times.append(deepcopy(rec_time))


    # save
    exp_path = os.path.join(fig_path, exp.name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Plot dist to reference
    plt.figure(figsize=(5, 3))
    dist_ref_ndarray = np.array(dist_ref)
    print("dist to ref:")
    for a_dist_ref in dist_ref_ndarray:
        print(a_dist_ref.mean(), "+-", a_dist_ref.std())

    plt.boxplot(dist_ref_ndarray.T)
    plt.xticks([1, 2, 3], labels)
    plt.xlabel('Strategy')
    plt.ylabel('Distance to strip center')
    fig_file = os.path.join(exp_path, f'distance_strip_{exp.name}_3_strats.pdf')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.show()

    # Plot success
    plt.figure(figsize=(5, 3))
    plt.bar([0, 1, 2], success)
    plt.xticks([0, 1, 2], labels)
    plt.xlabel('Strategy')
    plt.ylabel('Success rate [%]')
    fig_file = os.path.join(exp_path, f'success_rate_{exp.name}_3_strats.pdf')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.show()

    # print recovery time
    print("\nrecovery time:")
    for a_rec_time in rec_times:
        print(a_rec_time.mean(), "+-", a_rec_time.std())


