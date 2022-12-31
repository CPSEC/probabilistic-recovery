import os
import numpy as np
import matplotlib.pyplot as plt

recovery_folder = '/home/lin/workspace/probabilistic-recovery/src/recovery'
data_folder = os.path.join(recovery_folder, 'data')
state_file = os.path.join(data_folder, 'rxs.npz')

data = np.load(state_file)
rxs = data['rxs']
attack_index = data['attack_index']
recovery_index = data['recovery_index']
recovery_end_index = data['recovery_end_index']

print(f'{attack_index=},{recovery_index=},{recovery_end_index=}')

plt.rcParams.update({'font.size': 18})  # front size
fig = plt.figure(figsize=(10, 4))
s_index = attack_index - 10
e_index = recovery_end_index + 2

data_cnt = len(rxs)
e_cg = [val[0] for val in rxs[s_index: e_index]]
t = [val*0.05 for val in range(data_cnt)[s_index: e_index]]
ref = [0 for val in range(data_cnt)[s_index: e_index]]
plt.plot(t, ref, color='grey', linestyle='dashed')
plt.plot(t, e_cg)

# time
plt.axvline(attack_index*0.05, color='red', linestyle='dashed', linewidth=2)
plt.axvline((recovery_index-1)*0.05, color='green', linestyle='dotted', linewidth=2)

# strip
cnt = len(t)
y1 = [2.8]*cnt
y2 = [3.2]*cnt
plt.fill_between(t, y1, y2, facecolor='green', alpha=0.1)

# trim
plt.xlim((s_index*0.05, e_index*0.05))
plt.ylim((-4.4, 3.5))
plt.xlabel("Time [sec]")
plt.ylabel("Lateral Error [m]")

# save 
fig_file = os.path.join(data_folder, 'lgsvl_state.svg')
plt.savefig(fig_file, format='svg', bbox_inches='tight')

plt.show()
