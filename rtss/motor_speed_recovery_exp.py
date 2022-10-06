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
        'param': {'C': np.array([[0.1, 0], [0, 0.2]])}
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
    return data

k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(10)
print(k, P, arrive)

P_list = []
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(1)
print('k=', 1, '   P=', P, '   D_k=', D_k)
data = gaussian_data(D_k.sigma, D_k.miu)

P_list.append(P)
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(2)
print('k=', 2, '   P=', P, '   D_k=', D_k)
data1 = gaussian_data(D_k.sigma, D_k.miu)

P_list.append(P)
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(3)
print('k=', 3, '   P=', P, '   D_k=', D_k)
data2 = gaussian_data(D_k.sigma, D_k.miu)
mean = D_k.miu
cov = D_k.sigma
P_list.append(P)
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(4)
print('k=', 4, '   P=', P, '   D_k=', D_k)
data3 = gaussian_data(D_k.sigma, D_k.miu)

P_list.append(P)
X_k, D_k, z_star, alpha, P, arrive = reach.reachable_set_k(5)
print('k=', 5, '   P=', P, '   D_k=', D_k)
data4 = gaussian_data(D_k.sigma, D_k.miu)

P_list.append(P)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(data[:, 0], data[:, 1], c='blue', s=0.000005)
ax.scatter(data1[:, 0], data1[:, 1], c='blue', s=0.000005)
ax.scatter(data2[:, 0], data2[:, 1], c='blue', s=0.000005)
ax.scatter(data3[:, 0], data3[:, 1], c='blue', s=0.000005)
ax.scatter(data4[:, 0], data4[:, 1], c='blue', s=0.000005)
plt.text(4.382, 40.43, f'k={1} P={P_list[0]}')
plt.text(4.01, 38.79, f'k={2} P={P_list[1]}')
plt.text(3.95, 37.37, f'k={3} P={P_list[2]}')
plt.text(3.85, 35.87, f'k={4} P={P_list[3]}')
plt.text(3.75, 34.37, f'k={5} P={P_list[4]}')
plt.xlim(3.6,4.6)
plt.axvline(x=4.0,  c='green')
plt.axvline(x=3.8, linestyle='dashed', c='orange')
plt.axvline(x=4.2,   linestyle='dashed', c='orange')
ax.plot()
plt.show()
# plt.title(f'Covariance between x1 and x2 = {cov}')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.ylim(3.6, 4.4)
# plt.axis('equal')
# plt.show()

from scipy.stats.mvn import mvnun
res=mvnun(np.array([3.8, -np.inf]), np.array([4.2, np.inf]), mean, cov)
print(f"{res=}")