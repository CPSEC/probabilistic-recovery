import numpy as np
import matplotlib.pyplot as plt

data = np.load('x.npz')
x = data['x']
data_num = x.shape[0]

t = [0.05*val for val in range(data_num)]
y = x[:, 0]
ref = [0]*data_num

attack_start_index = 629
recovery_index = 649
recovery_complete_index = 722
dt = 0.05
y_lim = (-2.5, 0.1)
x_lim = 600 * dt

plt.rcParams.update({'font.size': 18})  # front size
fig = plt.figure(figsize=(8, 4))
# plt.title("lateral error")
plt.plot(t, ref, color='black', linestyle='dashed')
plt.plot(t, y)
plt.vlines(attack_start_index * dt, y_lim[0], y_lim[1], colors='red', linestyle='dashed')
plt.vlines(recovery_index * dt, y_lim[0], y_lim[1], colors='green', linestyle='dotted', linewidth=2.5)
plt.ylim(y_lim)
plt.xlim(x_lim, recovery_complete_index * dt)
plt.ylabel('lateral error (m)', fontsize=24)
# plt.show()
plt.savefig('lgsvl_error.pdf', format='pdf', bbox_inches='tight')

