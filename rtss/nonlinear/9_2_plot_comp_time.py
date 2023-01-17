import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os

from settings import cstr
from settings import svl
# exps = [cstr]
exps = [svl]

labels = ['OPRP', 'OPRP-CL', 'RTR-LQR', 'VS']
markers = ["o", 'h', 'D', '^']
colors = ["C0", "C4", "C1", "C2"]
sheets = ['time_oprp_ol', 'time_oprp_cl', 'time_lqr', 'time_vsr']
data_path = '../res/data'
fig_path = '../res/figs'

for exp in exps:
    plt.figure(figsize=(8, 3))
    plt.rcParams.update({'font.size': 15})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    times = []
    for s, sheet_name in enumerate(sheets):
        csv_path = os.path.join(data_path, exp.name, sheet_name + '.csv')
        time_data = pd.read_csv(csv_path)
        t = time_data[time_data['k'] >= exp.recovery_index]
        times.append(t['comp_time'].values*1000)

    # plot
    plt.figure(figsize=(5, 3))
    plt.boxplot(times)
    plt.xlabel("Strategy")
    plt.ylabel("Computation time [ms]")
    plt.ylim(exp.y_lim_time)
    plt.xticks([1, 2, 3, 4], labels)

    # save
    exp_path = os.path.join(fig_path, exp.name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    fig_file = os.path.join(exp_path, f'comp_time_{exp.name}.pdf')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.show()


