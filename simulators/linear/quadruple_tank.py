# Ref: K. H. Johansson, "The quadruple-tank process: a multivariable laboratory process with an adjustable zero,
# " in IEEE Transactions on Control Systems Technology, vol. 8, no. 3, pp. 456-465, May 2000, doi: 10.1109/87.845876.
# Ref: J. Giraldo, S. H. Kafash, J. Ruths and A. A. Cardenas, "DARIA: Designing Actuators to Resist Arbitrary Attacks
# Against Cyber-Physical Systems," 2020 IEEE European Symposium on Security and Privacy (EuroS&P), 2020, pp. 339-353,
# doi: 10.1109/EuroSP48549.2020.00029.

import numpy as np
import math
from utils.Simulator import Simulator
from utils.PID import PID

# two operating points
op = True  # True: have minimum phase characteristics
# False: have nonminimum-phase characteristics

# parameters
A1 = A3 = 28  # Ai: cross-section of Tank i;   unit:cm^2
A2 = A4 = 32
a1 = a3 = 0.071  # cross-section of the outlet hole;  unit:cm^2
a2 = a4 = 0.057
kc = 0.5  # measured level signals are and kc*h1 and kc*h2.
g = 981  # acceleration of gravity
# parameters for corresponding operating points
h1_0 = 12.4 if op else 12.6
h2_0 = 12.7 if op else 13.0
h3_0 = 1.8 if op else 4.8
h4_0 = 1.4 if op else 4.9
v1_0 = 3.00 if op else 3.15
v2_0 = 3.00 if op else 3.15
T1 = A1 / a1 * math.sqrt(2 * h1_0 / g)  # 62 if op else 63        # Ti = Ai/ai*sqrt(2*hi_0/g)
T2 = A2 / a2 * math.sqrt(2 * h2_0 / g)  # 90 if op else 91
T3 = A3 / a3 * math.sqrt(2 * h3_0 / g)  # 23 if op else 39
T4 = A4 / a4 * math.sqrt(2 * h4_0 / g)  # 30 if op else 56
k1 = 3.33 if op else 3.14  # The voltage applied to Pump is vi and the corresponding flow is ki*vi.
k2 = 3.35 if op else 3.29
gamma1 = 0.70 if op else 0.43  # The flow to Tank 1 is gamma1*k1*v1 and the flow to Tank 4 is (1-gamma1)*k1*v1
gamma2 = 0.60 if op else 0.34

# IO
IO = """
System Input:   v1, v2  -  input voltages to the pumps
System Output:  y1, y2  -  voltages from level measurement devices
System Model:
    xi := hi - hi_0
    ui := vi - vi_0   
"""

A = [[-1 / T1, 0, A3 / (A1 * T3), 0],
     [0, -1 / T2, 0, A4 / (A2 * T4)],
     [0, 0, -1 / T3, 0],
     [0, 0, 0, -1 / T4]]

B = [[gamma1 * k1 / A1, 0],
     [0, gamma2 * k2 / A2],
     [0, (1 - gamma2) * k2 / A3],
     [(1 - gamma1) * k1 / A4, 0]]

C = [[kc, 0, 0, 0],
     [0, kc, 0, 0]]

D = [[0, 0], [0, 0]]

x_0 = np.matrix([[0], [0], [0], [0]])

# control parameters
Kp1 = 3.0
Ki1 = 1/30
Kd1 = 0
Kp2 = 2.7
Ki2 = 1/40
Kd2 = 0


class Controller:
    def __init__(self, dt):
        self.pid1 = PID(Kp1, Ki1, Kd1, current_time=-dt)
        self.pid1.clear()
        self.pid1.setWindup(100)
        self.pid1.setSampleTime(dt)
        self.pid2 = PID(Kp2, Ki2, Kd2, current_time=-dt)
        self.pid2.clear()
        self.pid2.setWindup(100)
        self.pid2.setSampleTime(dt)

    def update(self, ref, feedback_value, current_time):
        self.pid1.SetPoint = ref[0, 0]
        self.pid1.update(feedback_value[0, 0], current_time)
        self.pid2.SetPoint = ref[1, 0]
        self.pid2.update(feedback_value[1, 0], current_time)
        return np.matrix([[self.pid1.output], [self.pid2.output]])


class QuadrupleTank(Simulator):
    def __init__(self, name, dt, max_index):
        super().__init__('Quadruple Tank ' + name, dt, max_index)
        self.linear(A, B, C, D)
        controller = Controller(dt)
        settings = {
            'init_state': x_0,
            'feedback_type': 'output',
            'controller': controller
        }
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 2000
    dt = 0.1
    ref = [np.matrix([[7], [7]])]*1001 + [np.matrix([[14], [12]])]*1000
    quadruple_tank = QuadrupleTank('test', dt, max_index)
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
    ref1 = [x[0, 0] for x in quadruple_tank.refs[:max_index + 1]]
    y1_arr = [x[0, 0] for x in quadruple_tank.outputs[:max_index + 1]]
    ax1.set_title('x1')
    ax1.plot(t_arr, y1_arr, t_arr, ref1)
    ref2 = [x[1, 0] for x in quadruple_tank.refs[:max_index + 1]]
    y2_arr = [x[1, 0] for x in quadruple_tank.outputs[:max_index + 1]]
    ax2.set_title('x2')
    ax2.plot(t_arr, y2_arr, t_arr, ref2)
    plt.show()
