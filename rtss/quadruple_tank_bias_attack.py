import numpy as np
from simulators.linear.quadruple_tank import QuadrupleTank
from utils.attack import Attack

# --------------- parameters  -----------------------------
max_index = 2000
dt = 0.1
ref = [np.array([7, 7])] * 1001 + [np.array([7, 7])] * 1000
noise = {
    'process': {
        'type': 'white',
        'param': {'C': np.eye(4) * 0.1}
    }
}
quadruple_tank = QuadrupleTank('test', dt, max_index, noise)
checkpoint_index = 18
attack_start_index = 20
bias = np.array([-2.0, 0])
bias_attack = Attack('bias', bias)
# --------------- end of parameters  -----------------------------


for i in range(0, max_index + 1):
    assert quadruple_tank.cur_index == i
    quadruple_tank.update_current_ref(ref[i])
    # attack here
    quadruple_tank.evolve()
# print results
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 1)
ax1 = ax[0]
ax2 = ax[1]
t_arr = np.linspace(0, 10, max_index + 1)
ref1 = [x[0] for x in quadruple_tank.refs[:max_index + 1]]
y1_arr = [x[0] for x in quadruple_tank.outputs[:max_index + 1]]
ax1.set_title('x1')
ax1.plot(t_arr, y1_arr, t_arr, ref1)
ref2 = [x[1] for x in quadruple_tank.refs[:max_index + 1]]
y2_arr = [x[1] for x in quadruple_tank.outputs[:max_index + 1]]
ax2.set_title('x2')
ax2.plot(t_arr, y2_arr, t_arr, ref2)

plt.show()