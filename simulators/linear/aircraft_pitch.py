import numpy as np

from utils import PID, Simulator

# system dynamics
A = [[-0.313, 56.7, 0],
     [-0.0139, -0.426, 0],
     [0, 56.7, 0]]
B = [[0.232], [0.0203], [0]]
C = [[0, 0, 1]]
D = [[0]]

x_0 = np.array([0.0, 0.0, 0.0])

# control parameters
KP = 1.13
KI = 0.0263
KD = 0
# KP = 20
# KI = 0.1
# KD = 1

class Controller:
    def __init__(self, dt):
        self.pid = PID(KP, KI, KD, current_time=-dt)
        self.pid.clear()
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        # self.pid.setControlLimit()

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid.set_reference(ref[0])
        cin = self.pid.update(feedback_value[0], current_time)
        return np.array([cin])

class AircraftPitch(Simulator):
     def __init__(self, name, dt, max_index, noise=None):
          super().__init__('Aircraft Pitch ' + name, dt, max_index)
          self.linear(A, B, C)
          controller = Controller(dt)
          settings = {
               'init_state': x_0,
               'feedback_type': 'output',
               'controller': controller
          }
          if noise:
               settings['noise'] = noise
          self.sim_init(settings)

if __name__ == "__main__":
    max_index = 1500
    dt = 0.02
    ref = [np.array([0.2])] * 1501
    noise = {
        'measurement': {
            'type': 'white',
            'param': np.array([1]) * 0.005
        }
    }
    aircraft_pitch = AircraftPitch('test', dt, max_index, noise)
    for i in range(0, max_index + 1):
        assert aircraft_pitch.cur_index == i
        aircraft_pitch.update_current_ref(ref[i])
        # attack here
        aircraft_pitch.evolve()
    # print results
    import matplotlib.pyplot as plt

    t_arr = np.linspace(0, 10, max_index + 1)
    ref = [x[0] for x in aircraft_pitch.refs[:max_index + 1]]
    y_arr = [x[0] for x in aircraft_pitch.outputs[:max_index + 1]]

    plt.plot(t_arr, y_arr, t_arr, ref)
    plt.show()