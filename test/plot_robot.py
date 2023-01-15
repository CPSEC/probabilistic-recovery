import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Notes
# Check how I got the timing for the last iteration
# In general check last iteration: how did I get the position etc.
plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sheets = ['time_ours_ol', 'time_ours_cl', 'time_emsoft', 'time_vs']
labels = ['OPRP', 'OPRP-CL', 'RTR-LQR', 'VS']
markers = ["o", 'h', 'D', '^']
colors = ["C0", "C4", "C1", "C2"]
mpc = [0, 0, 0, 0]
t_detection = 66 

times = []
for s,sheet_name in enumerate(sheets):
    data = pd.read_excel('./model2/data_summary.xlsx', sheet_name=sheet_name)
    if mpc[s]:
        t = data[data["k"] >= t_detection]
        times.append(t['comp_time'].values*1000)
    else:
        t = data[data["k"] == t_detection]
        times.append(t['comp_time'].values*1000)

# Plot times
plt.figure(figsize=(5, 3))
plt.boxplot(times)
plt.xlabel("Strategy")
plt.ylabel("Computation time [ms]")
plt.xticks([1, 2, 3, 4], labels)
plt.savefig("comp_time_traxxas_car_bp.pdf", bbox_inches='tight')
# Plot states

sheets = ['states_ours_ol', 'states_ours_cl', 'states_emsoft', 'states_vs']
plt.figure(figsize=(8, 3))

min_x = 0
max_x = 0
for s,sheet_name in enumerate(sheets):
    states_data = pd.read_excel('./model2/data_summary.xlsx', sheet_name=sheet_name)
    # print(states_data["pos_y"], states_data["pos_x"])
    x = states_data["pos_x"]
    y = states_data["pos_y"]
    plt.plot(x, y, color=colors[s], label=labels[s], linewidth=2)
    plt.plot(x[0], y[0], color=colors[s], marker='o', markersize=8)
    plt.plot(x[x.size - 1], y[y.size - 1], color=colors[s], marker='*', markersize=10)

    min_x = min(min_x, min(states_data["pos_x"]) )
    max_x = max(max_x, max(states_data["pos_x"]) )

axes = plt.gca()
xlim = axes.get_xlim()
x_values = np.array([min_x - 1, max_x + 1])
lane_width = 0.3
target_set_up = -0.3
target_set_lo = -0.7

# Plot nice figure
plt.plot(x_values, x_values * 0 + 3*lane_width, 'k', linewidth=2)
plt.plot(x_values, x_values * 0 + lane_width, 'y--', linewidth=2)
plt.plot(x_values, x_values * 0 - lane_width, 'k', linewidth=2)
plt.fill_between(x_values, x_values * 0 + target_set_up, x_values * 0 + target_set_lo, color='g', alpha=0.3)
plt.fill_between(x_values, x_values * 0 + 3*lane_width, x_values * 0 - lane_width, color='k', alpha=0.2)
axes = plt.gca()
ylim = axes.get_ylim()
plt.fill_between(x_values, x_values * 0 + 4*lane_width, x_values * 0 + lane_width*1.05, color='r', alpha=0.3)
plt.fill_between(x_values, x_values * 0 + target_set_lo, x_values * 0 + target_set_lo*2, color='r', alpha=0.3)
axes.set_ylim( ylim )
axes.set_xlim( [0.5, xlim[1]] )
plt.xlabel(r'$x$ position [m]')
plt.ylabel(r'$y$ position [m]')
plt.legend(ncol=2)
plt.savefig("timeseries_traxxas_car.pdf", bbox_inches='tight')



# Compute distance to strip
def compute_set_distance(xs, low, up):
    dists = []
    for x in xs:
        if x < low or x > up:
            dists.append(min( np.abs(x - low), np.abs(x - up) )) 
        else:
            dists.append(0)
    return dists
strategies = ['ours_ol', 'ours_cl', 'emsoft', 'virtual_sensors']
labels = ['OPRP', 'OPRP-CL', 'RTR-LQR', 'VS']
data = pd.read_excel('./model2/data_summary.xlsx', sheet_name='final_states')
success = np.zeros(len(strategies))
dist_ref = []
dist_set = []
for s, strategy in enumerate(strategies):
    data_strat = data[data["name"] == strategy ]
    
    success_exps = data_strat[(data_strat['final_pos_y'] <= target_set_up) & (data_strat['final_pos_y'] >= target_set_lo)]
    success[s] = len(success_exps) / 10 * 100
    dist = abs(-0.5 - data_strat['final_pos_y'])
    dist_ref.append(dist)

    
    dist_set.append(compute_set_distance(data_strat['final_pos_y'].values, target_set_lo, target_set_up))

# Plot dist to reference
plt.figure(figsize=(5, 3))
plt.boxplot(dist_ref)
plt.xticks([1, 2, 3, 4], labels)
plt.xlabel('Strategy')
plt.ylabel('Distance to reference [m]')
plt.savefig("distance_center_traxxas.pdf", bbox_inches='tight')

# Plot dist to strip
plt.figure(figsize=(5, 3))
plt.boxplot(dist_set)
plt.xticks([1, 2, 3, 4], labels)
plt.xlabel('Strategy')
plt.ylabel('Distance to shoulder [m]')
plt.savefig("distance_shoulder_traxxas.pdf", bbox_inches='tight')

# Plot success rate
plt.figure(figsize=(5, 3))
plt.bar([0, 1, 2, 3], success)
plt.xticks([0, 1, 2, 3], labels)
plt.xlabel('Strategy')
plt.ylabel('Success rate [%]')
plt.savefig("success_rate_traxxas.pdf", bbox_inches='tight')



plt.show()

