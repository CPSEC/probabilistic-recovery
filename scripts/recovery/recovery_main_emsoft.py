import numpy as np
from recovery.System import SystemModel
from recovery.utils.controllers.MPC import MPC
from recovery.utils.observers.full_state_bound import Estimator

class RecoveryEmsoft():
    def __init__(self, dt, u_min, u_max, attacked_sensor, isolation, noise):
        self.u_min = u_min
        self.u_max = u_max
        system_model = SystemModel(dt, u_min, u_max)
        self.system_model = system_model
        self.u_reconf = []

        self.estimator = Estimator(system_model.Ad, system_model.Bd, max_k = 150*2, epsilon= 2e-3 + noise)



        self.Q  = np.diag([1, 1, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0])
        self.Q[attacked_sensor, attacked_sensor] = 1

        self.QN = np.diag([1, 1, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0])
        self.QN[attacked_sensor, attacked_sensor] = 1

        self.R  = np.eye(system_model.m)/1000
        
        scale_lo = - 3000*noise
        scale_up = 3000*noise

        scale_h_lo = - 3000*noise
        scale_h_up = 300*noise
        #                               x1,             x2,               x3, v1, v2,  v3, r11, r12, r13, r21, r22, r23, r31, r32, r33, w1, w2, w3,
        safe_lo = np.array([-10 + scale_lo, -10 + scale_lo, -20 + scale_h_lo, -3, -3,  -3,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5, -1, -1, -1])
        safe_lo = safe_lo - system_model.x0

        #                               x1,             x2, x3, v1, v2, v3, r11, r12, r13, r21, r22, r23, r31, r32, r33, w1, w2, w3,
        safe_up = np.array([ 10 + scale_up,  10 + scale_up,  -8,  3,  3,  3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  1,  1,  1])
        safe_up = safe_up - system_model.x0


        self.safe_lo = safe_lo
        self.safe_up = safe_up

        #                      x1,  x2,   x3, v1,  v2, v3, r11, r12, r13, r21, r22, r23, r31, r32, r33, w1, w2, w3,
        target_lo = np.array([-5 + scale_lo, -5+ scale_lo, -15 + scale_h_lo, -3,  -3, -3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1])
        target_lo = target_lo - system_model.x0
        self.target_lo = target_lo

        #                     x1, x2,   x3, v1, v2, v3, r11, r12, r13, r21, r22, r23, r31, r32, r33, w1, w2, w3,
        target_up = np.array([5 + scale_up, 5 + scale_up, -9.5 + scale_h_up,  3,  3,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,  1,  1,  1])
        target_up = target_up - system_model.x0
        self.target_up = target_up

        #                x1, x2, x3, v1, v2, v3, r11, r12, r13, r21, r22, r23, r31, r32, r33, w1, w2, w3,
        ref = np.array([0,  0, -10,  0,  0,  0,   1,   0,   0,   0,   1,   0,   0,   0,   1,  0,  0,  0])
        ref = ref - system_model.x0
        self.ref = ref
        

    

    def init_open_loop(self):
        self.x_checkpoint = []
        self.u_checkpoint = []
    
    def init_closed_loop(self, C):
        pass
        

    def checkpoint_state(self, state):
        x = state
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
        x_cur_lo, x_cur_up, x_curr = self.estimator.estimate(self.x_checkpoint, self.u_checkpoint)
        control = self.u_checkpoint[-1]
        k = self.estimator.get_deadline(x_curr, self.safe_lo, self.safe_up, control, 100)
        
        mpc_settings = {
            'Ad': self.system_model.Ad, 'Bd': self.system_model.Bd,
            'Q': self.Q, 'QN': self.QN, 'R':self.R,
            'N': k+3,
            'ddl': k, 'target_lo':self.target_lo, 'target_up':self.target_up,
            'safe_lo':self.safe_lo, 'safe_up':self.safe_up,
            'control_lo': self.u_min, 'control_up': self.u_max,
            'ref':self.ref
        }
        self.mpc = MPC(mpc_settings)

        self.k_max = k + 3
        _, rec_u = self.mpc.update(feedback_value= x_curr)
        fM = rec_u[0] # Pick the first u
        self.k_recovery = 0
        
        self.u_reconf = rec_u
        fM = fM + self.system_model.u0
        # Gazebo uses a different frame
        fM = self.convert_input(fM)
        
        print("number of recovery steps: ", self.k_max)
        return fM, self.k_max
    
    # Auxiliar function to call the recovery 
    def update_recovery_ni(self, state, u):
        fM = np.zeros((4,1))
        
        fM = self.u_reconf[self.k_recovery] # look up vector
        self.k_max -= 1
        self.k_recovery += 1
        # print(fM)

        # Gazebo uses a different frame

        fM = fM + self.system_model.u0
        fM = self.convert_input(fM)
        return fM, self.k_max

    def convert_input(self, fM):
        fM[2] = -fM[2]
        fM[3] = -fM[3]
        return fM

    # Auxiliary function to flatten the state vector
    # def process_state(self, x):
    #     pos = x[0]
    #     v = x[1]
    #     R = x[3].T
    #     R = R.flatten()
    #     w = x[4]
    #     x = np.concatenate((pos, v, R, w)).flatten()
    #     return x


        
    

    
    
