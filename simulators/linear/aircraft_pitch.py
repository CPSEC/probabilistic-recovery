#Ref: https://ctms.engin.umich.edu/CTMS/index.php?example=AircraftPitch&section=ControlPID
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
KI = 0.0253
KD = 0.0
control_limit = {'lo': [-20], 'up': [20]}

class Controller:
    def __init__(self, dt):
        self.dt = dt
        self.pid = PID(KP, KI, KD, current_time=-dt)
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        self.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid.set_reference(ref[0])
        cin = self.pid.update(feedback_value[0], current_time)
        return np.array([cin])

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.pid.set_control_limit(self.control_lo[0], self.control_up[0])

    def clear(self):
        self.pid.clear(current_time=-self.dt)


class AircraftPitch(Simulator):
    """
      States: (3,)
          x[0]: Angle of attack
          x[1]: Pitch rate
          x[2]: Pitch angle
      Control Input: (1,)
          u[0]: the elevator deflection angle
      Output:  (1,)
          y[0]: the pitch angle of the aircraft
          Output Feedback
      Controller: PID
      """
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
        'process': {
            'type': 'white',
            'param': {'C': np.eye(3) * 0.00001}
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

    u_arr = [x[0] for x in aircraft_pitch.inputs[:max_index + 1]]
    plt.plot(t_arr, u_arr)
