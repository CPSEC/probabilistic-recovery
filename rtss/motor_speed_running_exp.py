from simulators import MotorSpeed
import numpy as np

max_index = 500
dt = 0.02
ref = [np.array([4])] * 501
noise = {
    'measurement': {
        'type': 'white',
        'param': np.array([1]) * 0.05
    }
}
motor_speed = MotorSpeed('test', dt, max_index, noise)
for i in range(0, max_index + 1):
    assert motor_speed.cur_index == i
    motor_speed.update_current_ref(ref[i])
    # attack here
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