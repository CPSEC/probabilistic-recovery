from simulators import MotorSpeed
import numpy as np
from utils.attack import Attack

# --------------- parameters  -----------------------------
max_index = 500
dt = 0.02
ref = [np.array([4])] * 501
noise = {
    'process': {
        'type': 'white',
        'param': {'C': np.array([[0.08, 0], [0, 0.001]])}
    }
}
motor_speed = MotorSpeed('test', dt, max_index, noise)
attack_start_index = 150
bias = np.array([-0.3])
bias_attack = Attack('bias', bias, attack_start_index)
recovery_index = 200
# --------------- end of parameters -------------------------



for i in range(0, max_index + 1):
    assert motor_speed.cur_index == i
    motor_speed.update_current_ref(ref[i])
    # attack here
    motor_speed.cur_feedback = bias_attack.launch(motor_speed.cur_feedback, i, motor_speed.states)
    if i == recovery_index:
        print('recovery_start_state=', motor_speed.cur_x)
    motor_speed.evolve()



# print results
import matplotlib.pyplot as plt

t_arr = np.linspace(0, 10, max_index + 1)
ref = [x[0] for x in motor_speed.refs[:max_index + 1]]
y_arr = [x[0] for x in motor_speed.outputs[:max_index + 1]]

plt.plot(t_arr, y_arr, t_arr, ref)
plt.show()

u_arr = [x[0] for x in motor_speed.inputs[:max_index + 1]]
plt.plot(t_arr, u_arr)
plt.show()



# -----------         output -------------------------
# recovery_start_state= [ 4.4392062  42.17318198]