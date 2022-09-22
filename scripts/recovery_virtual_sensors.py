import numpy as np
from recovery.System import System
from recovery.utils.observers.full_state_bound import Estimator

class RecoveryVirtualSensor():
    def __init__(self, dt, u_min, u_max, attacked_sensor, isolation):
        self.u_min = u_min
        self.u_max = u_max
        system_model = System(dt, u_min, u_max)
        self.system_model = system_model

        self.estimator = Estimator(system_model.Ad, system_model.Bd, max_k = 150, epsilon= 1e-7)
    
    def init_open_loop(self):
        self.x_checkpoint = []
        self.u_checkpoint = []
    
    def init_closed_loop(self, C):
        pass
        

    def checkpoint_state(self, state):
        x = self.process_state(state)
        self.x_checkpoint = x - self.system_model.x0

    def checkpoint(self, x, u):
        u = u.flatten()
        du = u - self.system_model.u0
        # x = self.process_state(x)
        self.checkpoint_input_open_loop(du)
    
    # Checkpoint the input for the open loop strategy
    def checkpoint_input_open_loop(self, du_0):
        self.u_checkpoint.append(du_0)

    # Checkpoint the input and measurement for the closed loop strategy
    def checkpoint_closed_loop(self, dy, du):
        pass

    # Auxiliar function to call the recovery for the first time
    def update_recovery_fi(self):
        x_cur_lo, x_cur_up, x_cur = self.estimator.estimate(self.x_checkpoint, self.u_checkpoint)
        x_cur = x_cur + self.system_model.x0
        y = self.system_model.Cd @ x_cur
        return self.pack_state(y.flatten())


    def update_recovery_ni(self, x, u):
        x = self.process_state(x) - self.system_model.x0
        u = u.flatten() - self.system_model.u0

        x_cur_lo, x_cur_up, x_cur = self.estimator.estimate(x, [u])
        x_cur = x_cur + self.system_model.x0
        y = self.system_model.Cd @ x_cur
        return self.pack_state(y.flatten())
    
    def pack_state(self, state):
        pos = state[0:3]
        vel = state[3:6]
        R_aux = state[6:15]
        w = state[15:]
        R = np.zeros((3, 3))
        for i in range(0, 3):
            R[i, :] = R_aux[i*3: (i+1)*3]
        R = R.T
        states = (pos, vel, vel*0, R, w)
        return states

    # Auxiliary function to flatten the state vector
    def process_state(self, x):
        pos = x[0]
        v = x[1]
        R = x[3].T
        R = R.flatten()
        w = x[4]
        x = np.concatenate((pos, v, R, w)).flatten()
        return x

