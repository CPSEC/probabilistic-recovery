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
sheets = ['states_oprp_ol', 'states_oprp_cl', 'states_lqr', 'states_vsr']
data_path = '../res/data'
fig_path = '../res/figs'

for exp in exps:
    plt.figure(figsize=(8, 3))
    plt.rcParams.update({'font.size': 15})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    for s, sheet_name in enumerate(sheets):
        csv_path = os.path.join(data_path, exp.name, sheet_name + '.csv')
        states_data = pd.read_csv(csv_path)
        x = states_data["time"]
        y = states_data[exp.state_names[exp.output_index]]
        # plot data curve
        plt.plot(x, y, color=colors[s], label=labels[s], linewidth=2)
        plt.plot(x[0], y[0], color=colors[s], marker='o', markersize=8)
        plt.plot(x[x.size - 1], y[y.size - 1], color=colors[s], marker='*', markersize=10)

    # strip
    x_range = np.array(exp.x_lim)
    y1 = exp.strip[0]
    y2 = exp.strip[1]
    plt.fill_between(x_range, y1, y2, facecolor='green', alpha=0.3)

    # plot when attack starts and whe recovery starts
    # plt.vlines(exp.attack_start_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed',
               # linewidth=2)
    # plt.vlines(exp.recovery_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted',
    #            linewidth=2)
    plt.vlines(x[exp.attack_start_index], exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed',
               linewidth=2)
    plt.vlines(x[exp.recovery_index], exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted',
               linewidth=2)

    # zoom in
    if exp.y_lim:
        plt.ylim(exp.y_lim)
    if exp.x_lim:
        plt.xlim(exp.x_lim)

    # labels and legends
    plt.ylabel(exp.y_label)
    plt.xlabel('Time [sec]')
    plt.legend(ncol=2, loc=exp.legend_loc)

    # save
    exp_path = os.path.join(fig_path, exp.name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    fig_file = os.path.join(exp_path, f'timeseries_{exp.name}.pdf')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.show()


# min_x = 0
# max_x = 0
# for s,sheet_name in enumerate(sheets):
#     states_data = pd.read_excel('./model2/data_summary.xlsx', sheet_name=sheet_name)
#     # print(states_data["pos_y"], states_data["pos_x"])
#     x = states_data["pos_x"]
#     y = states_data["pos_y"]
#     plt.plot(x, y, color=colors[s], label=labels[s], linewidth=2)
#     plt.plot(x[0], y[0], color=colors[s], marker='o', markersize=8)
#     plt.plot(x[x.size - 1], y[y.size - 1], color=colors[s], marker='*', markersize=10)
#
#
#
#
#
#     min_x = min(min_x, min(states_data["pos_x"]) )
#     max_x = max(max_x, max(states_data["pos_x"]) )
#
# axes = plt.gca()
# xlim = axes.get_xlim()
# x_values = np.array([min_x - 1, max_x + 1])
# lane_width = 0.3
# target_set_up = -0.3
# target_set_lo = -0.7
#
# # Plot nice figure
# plt.plot(x_values, x_values * 0 + 3*lane_width, 'k', linewidth=2)
# plt.plot(x_values, x_values * 0 + lane_width, 'y--', linewidth=2)
# plt.plot(x_values, x_values * 0 - lane_width, 'k', linewidth=2)
# plt.fill_between(x_values, x_values * 0 + target_set_up, x_values * 0 + target_set_lo, color='g', alpha=0.3)
# plt.fill_between(x_values, x_values * 0 + 3*lane_width, x_values * 0 - lane_width, color='k', alpha=0.2)
# axes = plt.gca()
# ylim = axes.get_ylim()
# plt.fill_between(x_values, x_values * 0 + 4*lane_width, x_values * 0 + lane_width*1.05, color='r', alpha=0.3)
# plt.fill_between(x_values, x_values * 0 + target_set_lo, x_values * 0 + target_set_lo*2, color='r', alpha=0.3)
# axes.set_ylim( ylim )
# axes.set_xlim( [0.5, xlim[1]] )
# plt.xlabel(r'$x$ position [m]')
# plt.ylabel(r'$y$ position [m]')
# plt.legend(ncol=2)
# plt.savefig("timeseries_traxxas_car.pdf", bbox_inches='tight')