from simulators import MotorSpeed
import numpy as np
from utils.attack import Attack

from utils.formal import Zonotope
from utils.formal.reachability import ReachableSet
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.strip import Strip
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from scipy.stats import kde
# --------------- parameters  -----------------------------
max_index = 500
dt = 0.02
ref = [np.array([4])] * 501
noise = {
    'process': {
        'type': 'white',
        'param': {'C': np.array([[0.07, 0], [0, 0.2]])}
    }
}
motor_speed = MotorSpeed('test', dt, max_index, noise)
attack_start_index = 150
bias = np.array([-1.0])
bias_attack = Attack('bias', bias, attack_start_index)
recovery_index = 200
# --------------- end of parameters -------------------------

u_lo = motor_speed.controller.control_lo
u_up = motor_speed.controller.control_up
U = Zonotope.from_box(u_lo, u_up)
print(U)

A = motor_speed.sysd.A
B = motor_speed.sysd.B
W = motor_speed.p_noise_dist
reach = ReachableSet(A, B, U, W, max_step=50)

x_0 = GaussianDistribution(np.array([4.4392062,  42.17318198]), np.zeros((2, 2)))
s = Strip(np.array([-1, 0]), a=-4.2, b=-3.8)
reach.init(x_0, s)

# fig_setting = {'x1': 0, 'x2': 80, 'y1': 0, 'y2': 90,
#                'strip': True, 'routine': True,
#                'zonotope': True, 'distribution': True}
fig_setting = {'x1': 3.5, 'x2': 4.5, 'y1': 30, 'y2': 48,
               'strip': False, 'routine': True,
               'zonotope': True, 'distribution': True,
               'head_width': 0.01, 'width': 0.002}
# X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(2)
# # reach.plot(X_k, D_k, alpha, fig_setting)
#
# X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(10)
# # reach.plot(X_k, D_k, alpha, fig_setting)
# print(P)
#
# i, satisfy, X_k, D_k, z_star, alpha, P, arrive = reach.given_P(P_given=0.95, max_k=40)
# print('i=', i, 'found=', satisfy, 'P=', P)
# reach.plot(X_k, D_k, alpha, fig_setting)

def gaussian_data(sigma, miu):
    cov = D_k.sigma
    mean = D_k.miu
    random_seed = 1000
    distr = multivariate_normal(cov=cov, mean=mean,
                                seed=random_seed)
    data = distr.rvs(size=200000)
    return data, mean

k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(10)
print(k, P, arrive)

P_list = []
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(1)
print('k=', 1, '   P=', P, '   D_k=', D_k)
data, mean = gaussian_data(D_k.sigma, D_k.miu)
P_list.append(P)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(2)
print('k=', 2, '   P=', P, '   D_k=', D_k)
data1, mean1 = gaussian_data(D_k.sigma, D_k.miu)
P_list.append(P)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(3)
print('k=', 3, '   P=', P, '   D_k=', D_k)
data2, mean2 = gaussian_data(D_k.sigma, D_k.miu)
P_list.append(P)

X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(4)
print('k=', 4, '   P=', P, '   D_k=', D_k)
data3, mean3 = gaussian_data(D_k.sigma, D_k.miu)

P_list.append(P)
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(5)
print('k=', 5, '   P=', P, '   D_k=', D_k)
data4, mean4 = gaussian_data(D_k.sigma, D_k.miu)

P_list.append(P)

fig = plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 14})
# ax = fig.add_subplot(111)

plt.axvline(x=4.0,  linestyle='dashed', c='grey')
plt.axvline(x=3.8, linestyle='dashed', c='green')
plt.axvline(x=4.2,   linestyle='dashed', c='green')
plt.axvspan(3.8, 4.2, alpha=0.1, color='green')

plt.scatter(data[:, 0], data[:, 1], c='blue', s=0.000005)
plt.scatter(data1[:, 0], data1[:, 1], c='blue', s=0.000005)
plt.scatter(data2[:, 0], data2[:, 1], c='blue', s=0.000005)
plt.scatter(data3[:, 0], data3[:, 1], c='blue', s=0.000005)
plt.scatter(data4[:, 0], data4[:, 1], c='blue', s=0.000005)
plt.text(3.92, mean[1]-0.15, f'k={1}, P={float(P_list[0]):.3f}', fontsize=14)
plt.text(3.87, mean1[1]-0.15, f'k={2}, P={float(P_list[1]):.3f}', fontsize=14)
plt.text(3.75, mean2[1]-0.15, f'k={3}, P={float(P_list[2]):.3f}', fontsize=14)
plt.text(4.3, mean3[1]-0.15, f'k={4}, P={float(P_list[3]):.3f}', fontsize=14)
plt.text(4.22, mean4[1]-0.15, f'k={5}, P={float(P_list[4]):.3f}', fontsize=14)
plt.xlim(3.7, 4.6)
plt.ylim(33, 41.5)

plt.xlabel('Rotational Speed $x_1$ (rad/sec)')
plt.ylabel('Electric Current $x_2$ (Amp)')

plt.plot()
plt.savefig(f'fig/baselines/2D_distribution.png', bbox_inches='tight')
plt.show()
# plt.title(f'Covariance between x1 and x2 = {cov}')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.ylim(3.6, 4.4)
# plt.axis('equal')
# plt.show()

# from scipy.stats.mvn import mvnun
# res=mvnun(np.array([3.8, -np.inf]), np.array([4.2, np.inf]), mean, cov)
# print(f"{res=}")