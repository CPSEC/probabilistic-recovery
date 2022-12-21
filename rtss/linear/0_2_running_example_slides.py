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

x_0 = GaussianDistribution(np.array([4.4392062, 42.17318198]), np.zeros((2, 2)))
s = Strip(np.array([-1, 0]), a=-4.2, b=-3.8)
reach.init(x_0, s)

# fig_setting = {'x1': 0, 'x2': 80, 'y1': 0, 'y2': 90,
#                'strip': True, 'routine': True,
#                'zonotope': True, 'distribution': True}
fig_setting = {'x1': 3.5, 'x2': 4.5, 'y1': 30, 'y2': 48,
               'strip': False, 'routine': True,
               'zonotope': True, 'distribution': True,
               'head_width': 0.01, 'width': 0.002}


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

k = [1, 2, 3, 4, 5]
P_list = []
X_list = []
point_list = []
mean_list = []
label_x_loc = [3.92, 3.87, 3.75, 4.3, 4.22]

for i in range(5):
    X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(k[i])
    data, mean = gaussian_data(D_k.sigma, D_k.miu)
    P_list.append(P)
    X_list.append(X_k)
    point_list.append(data)
    mean_list.append(mean)

for cnt in range(5):

    fig = plt.figure(figsize=(6, 4))
    plt.rcParams.update({'font.size': 14})
    # ax = fig.add_subplot(111)

    plt.axvline(x=4.0, linestyle='dashed', c='grey')
    plt.axvline(x=3.8, linestyle='dashed', c='green')
    plt.axvline(x=4.2, linestyle='dashed', c='green')
    plt.axvspan(3.8, 4.2, alpha=0.1, color='green')

    for i in range(cnt + 1):
        plt.scatter(point_list[i][:, 0], point_list[i][:, 1], c='blue', s=0.000005)
        plt.text(label_x_loc[i], mean_list[i][1] - 0.15, f'k={k[i]}, P={float(P_list[i]):.3f}', fontsize=14)

    plt.xlim(3.7, 4.6)
    plt.ylim(33, 41.5)

    plt.xlabel('Rotational Speed $x_1$ (rad/sec)')
    plt.ylabel('Electric Current $x_2$ (Amp)')

    plt.plot()
    plt.savefig(f'fig/slides/{k[cnt]}.png', bbox_inches='tight')
    # plt.show()

for cnt in range(1, 5):
    fig = plt.figure(figsize=(6, 4))
    plt.rcParams.update({'font.size': 14})
    # ax = fig.add_subplot(111)

    plt.axvline(x=4.0, linestyle='dashed', c='grey')
    plt.axvline(x=3.8, linestyle='dashed', c='green')
    plt.axvline(x=4.2, linestyle='dashed', c='green')
    plt.axvspan(3.8, 4.2, alpha=0.1, color='green')

    for i in range(cnt):
        plt.scatter(point_list[i][:, 0], point_list[i][:, 1], c='blue', s=0.000005)
        plt.text(label_x_loc[i], mean_list[i][1] - 0.15, f'k={k[i]}, P={float(P_list[i]):.3f}', fontsize=14)

    if 0 < cnt < 5:
        X_list[cnt].plot(fig)
        plt.plot(mean_list[cnt][0], mean_list[cnt][1], marker='o')

    plt.xlim(3.7, 4.6)
    plt.ylim(33, 41.5)

    plt.xlabel('Rotational Speed $x_1$ (rad/sec)')
    plt.ylabel('Electric Current $x_2$ (Amp)')

    plt.plot()
    plt.savefig(f'fig/slides/{k[cnt]}_zonotope.png', bbox_inches='tight')
