import numpy as np
from recovery.System import System
from recovery.rtss import RTSS
from recovery.utils.formal.gaussian_distribution import GaussianDistribution

class Recovery():
    def __init__(self, dt, u_min, u_max, attacked_sensor, isolation):
        system_model = System(dt, u_min, u_max)
        self.system_model = system_model
        self.W = 0.001 * np.eye(system_model.n)
        mu = np.zeros((self.system_model.n))
        self.W = GaussianDistribution.from_standard(mu, self.W)

        l = np.array([0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0])
        l[attacked_sensor] = 1
        a = -0.1
        b = 0.1

        self.rtss = RTSS(system_model.Ad, system_model.Bd, system_model.Cd, system_model.Dd, self.W, u_min, u_max, k_reconstruction=500, k_max=20, l=l, a=a, b=b )
        self.isolation = isolation
        
        self.k_recovery = -1
        self.u_reconf = []
        self.k_max = -1
    

    def init_open_loop(self):
        self.x_checkpoint = []
        self.u_checkpoint = []
    
    def init_closed_loop(self, C):
        self.x_checkpoint = []
        self.u_checkpoint = []
        self.y_checkpoint = []
        self.C_kf = C
        R = np.eye(C.shape[0]) *0.1
        self.rtss.set_kalman_filter(C, self.W.sigma, R)
        

    def checkpoint_state(self, state):
        x = self.process_state(state)
        self.x_checkpoint = x - self.system_model.x0

    def checkpoint(self, x, u):
        u = u.flatten()
        du = u - self.system_model.u0
        x = self.process_state(x)
        if self.isolation:
            dy = self.C_kf @ (x - self.system_model.x0)
            self.checkpoint_closed_loop(dy, du)
        else:
            self.checkpoint_input_open_loop(du)
    
    # Checkpoint the input for the open loop strategy
    def checkpoint_input_open_loop(self, du_0):
        self.u_checkpoint.append(du_0)

    # Checkpoint the input and measurement for the closed loop strategy
    def checkpoint_closed_loop(self, dy, du):
        self.y_checkpoint.append(dy)
        self.u_checkpoint.append(du)
        pass

    # Auxiliar function to call the recovery for the first time
    def update_recovery_fi(self):
        fM = np.zeros((4,1))
        self.k_max = 0
        if self.isolation:
            self.u_reconf, self.k_max = self.rtss.recovery_isolation_fs(self.u_checkpoint, self.y_checkpoint, self.x_checkpoint)
            fM = self.u_reconf[0] # receding horizon control
        else:
            self.u_reconf, self.k_max = self.rtss.recovery_no_isolation(self.u_checkpoint, self.x_checkpoint)
            fM = self.u_reconf[0] # Take the first u
        self.k_recovery = 0

        # Gazebo uses a different frame
        fM = self.convert_input(fM)

        fM = fM + self.system_model.u0
        print("number of recovery steps: ", self.k_max)
        return fM, self.k_max
    
    # Auxiliar function to call the recovery 
    def update_recovery_ni(self, state, u):
        fM = np.zeros((4,1))
        
        
        x = self.process_state(state)
        u = u.flatten()

        dx = x - self.system_model.x0
        du = u - self.system_model.u0
        self.k_recovery += 1
        if self.isolation:
            fM, self.k_max = self.rtss.recovery_isolation_ns(dx, du)
        else:
            fM = self.u_reconf[self.k_recovery] # look up vector
            self.k_max -= 1
        # print(fM)

        # Gazebo uses a different frame
        fM = self.convert_input(fM)

        fM = fM + self.system_model.u0
        return fM, self.k_max

    def convert_input(self, fM):
        fM[2] = -fM[2]
        fM[3] = -fM[3]
        return fM

    # Auxiliary function to flatten the state vector
    def process_state(self, x):
        pos = x[0]
        v = x[1]
        R = x[3].T
        R = R.flatten()
        w = x[4]
        x = np.concatenate((pos, v, R, w)).flatten()
        return x


        
    

    
    
