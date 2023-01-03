import numpy as np
import os, time
from scipy.signal import StateSpace
from scipy.integrate import solve_ivp
from utils.formal.gaussian_distribution import GaussianDistribution

rseed_str = os.getenv('RANDOM_SEED')
if rseed_str is None:
    rseed = np.uint32(int(time.time()))
else:
    try:
        rseed = np.uint32(rseed_str)
    except Exception as e:
        print('rseed read error:', e)
        rseed = np.uint32(int(time.time()))
print('rseed=', rseed)
np.random.seed(rseed)

class Simulator:
    """
    states, control inputs/outputs are instance of np.array with shape (n,) (m,) (p,)
    """

    def __init__(self, name, Ts, max_index):
        self.name = name
        self.model_type = None
        self.dt = Ts
        self.sysc = None
        self.sysd = None
        self.max_index = max_index
        self.C = None
        self.D = None
        self.n = None  # number of states
        self.m = None  # number of control inputs
        self.p = None  # number of sensor measurements
        self.ode = None  # ode function
        self.init_state = None
        # values under self.cur_index
        self.cur_x = None
        self.cur_y = None
        self.cur_u = None
        self.cur_feedback = None
        self.cur_ref = None
        self.cur_index = 0
        self.noise_setting = None
        self.m_noise = None
        self.p_noise = None
        self.m_noise_dist = None
        self.p_noise_dist = None

    def data_init(self):
        self.inputs = np.empty((self.max_index + 2, self.m), dtype=np.float)
        self.outputs = np.empty((self.max_index + 2, self.p), dtype=np.float)
        self.states = np.empty((self.max_index + 2, self.n), dtype=np.float)
        if self.feedback_type == 'output':
            self.feedbacks = np.empty((self.max_index + 2, self.p), dtype=np.float)
            self.refs = np.empty((self.max_index + 2, self.p), dtype=np.float)
        elif self.feedback_type == 'state':
            self.feedbacks = np.empty((self.max_index + 2, self.n), dtype=np.float)
            self.refs = np.empty((self.max_index + 2, self.n), dtype=np.float)  # reference value

    def reset(self):
        # the noise will not reset!
        self.data_init()
        self.cur_index = 0
        self.set_init_state(self.init_state)
        self.controller.clear()
    
    def reset_seed(self):
        rseed_str = os.getenv('RANDOM_SEED')
        if rseed_str is None:
            rseed = np.uint32(int(time.time()))
        else:
            try:
                rseed = np.uint32(rseed_str)
            except Exception as e:
                print('rseed read error:', e)
                rseed = np.uint32(int(time.time()))
        print('rseed=', rseed)
        np.random.seed(rseed)

    def linear(self, A, B, C=None, D=None, x_linear=None, u_linear=None):
        self.model_type = 'linear'
        if C is None:
            C = np.eye(len(A))
        if D is None:
            D = np.zeros((len(C), len(B[0])))
        if x_linear is None:
            x_linear = np.zeros((len(A), 1))
        if u_linear is None:
            u_linear = np.zeros((len(B), 1))
        self.sysc = StateSpace(A, B, C, D)
        self.sysd = self.sysc.to_discrete(self.dt)
        self.n = self.sysc.A.shape[1]
        self.m = self.sysc.B.shape[1]
        self.p = self.sysc.C.shape[0]
        self.C = self.sysc.C
        self.D = self.sysc.D
        self.x_linear = x_linear
        self.u_linear = u_linear

        def fprime(t, x, u):
            return self.sysc.A @ x + self.sysc.B @ u

        self.ode = fprime

    def nonlinear(self, ode, n, m, p, x_linear, u_linear, C=None, D=None):
        self.model_type = 'nonlinear'
        self.C = np.eye(n) if C is None else np.array(C)
        self.D = np.zeros((p, m)) if D is None else np.array(D)
        self.n = n
        self.m = m
        self.p = p
        self.ode = ode
        self.x_linear = x_linear
        self.u_linear = u_linear

    def sim_init(self, settings: dict):
        """
        keys:
          'feedback_type': 'state', 'output', None
          'init_state': np.ndarray   (n,)
          'controller':  object with update method
        """
        self.set_feedback_type(settings['feedback_type'])
        self.data_init()
        if 'noise' in settings:
            self.noise_init(settings['noise'])
        self.set_init_state(settings['init_state'])
        self.set_controller(settings['controller'])

    def noise_init(self, noise):
        """
        Only implement the white noise
        keys:
          'process'/'measurement':
            'type': 'white'  todo: 'white_bounded', 'box_uniform', 'ball_uniform'
            'param':
              'C': linear transformation matrix from standard normal distribution      scale for 'white'
        """
        if 'process' in noise:
            if noise['process']['type'] == 'white':
                miu = np.zeros((self.n,))
                C = noise['process']['param']['C']
                self.p_noise_dist = GaussianDistribution.from_standard(miu, C)
                self.p_noise = self.p_noise_dist.random(self.max_index + 2).T
        if 'measurement' in noise:
            if noise['measurement']['type'] == 'white':
                miu = np.zeros((self.p,))
                C = noise['measurement']['param']['C']
                self.m_noise_dist = GaussianDistribution.from_standard(miu, C)
                self.m_noise = self.m_noise_dist.random(self.max_index + 2).T

    def set_init_state(self, x):
        self.init_state = x
        self.cur_x = x
        self.cur_y = self.C @ self.cur_x
        if self.m_noise is not None:
            self.cur_y += self.m_noise[self.cur_index]
        self.outputs[0] = self.cur_y
        self.states[0] = self.cur_x
        if self.feedback_type:
            self.cur_feedback = self.cur_x if self.feedback_type == 'state' else self.cur_y

    def set_feedback_type(self, feedback_type):
        """
        'state', 'output', None
        """
        self.feedback_type = feedback_type

    def set_controller(self, controller):
        """
        please implement update method to get control input
        """
        self.controller = controller

    def update_current_ref(self, ref):
        self.cur_ref = ref

    # remove this
    def solve(self, x, u, dt):
        steps = 100/dt
        for _ in range( int(steps) ):
            dx = np.array( self.ode(dt, x, u) ) 
            x = x + dx / steps
            x = np.array([max(y, 0) for y in x])
        return x

    def evolve(self, u=None):
        # record data
        self.feedbacks[self.cur_index] = self.cur_feedback
        self.refs[self.cur_index] = self.cur_ref

        # compute control input
        if self.feedback_type:
            self.cur_u = self.controller.update(self.cur_ref, self.cur_feedback, self.dt * self.cur_index)
        else:
            self.cur_u = u
        # override control input
        if not (u is None):
            self.cur_u = u
        assert self.cur_u.shape == (self.m,)
        self.inputs[self.cur_index] = self.cur_u

        # implement control input
        ts = (self.cur_index * self.dt, (self.cur_index + 1) * self.dt)
        # res = solve_ivp(self.ode, ts, self.cur_x, args=(self.cur_u,))
        self.cur_index += 1
        # self.cur_x = res.y[:, -1]
        if self.model_type == 'linear':
            self.cur_x = self.sysd.A @ self.cur_x + self.sysd.B @ self.cur_u
        else:
            cur_u = self.cur_u + self.u_linear
            cur_x = self.cur_x + self.x_linear
            # Modify this. It is too dirty.
            # TODO: add new element to super class regarding the states boundaries
            if self.name == 'Quadruple Tank test':
                res = self.solve(cur_x, cur_u, self.dt)
                self.cur_x = res - self.x_linear
            else:
                res = solve_ivp(self.ode, ts, cur_x, args=(cur_u, ), max_step = self.dt/10)
                self.cur_x = res.y[:, -1] - self.x_linear
        if self.p_noise is not None:  # process noise
            self.cur_x += self.p_noise[self.cur_index]
        self.cur_y = self.C @ self.cur_x + self.D @ self.cur_u
        if self.m_noise is not None:  # measurement noise
            self.cur_y += self.m_noise[self.cur_index]
        assert self.cur_x.shape == (self.n,)
        assert self.cur_y.shape == (self.p,)
        self.states[self.cur_index] = self.cur_x
        self.outputs[self.cur_index] = self.cur_y

        # prepare feedback
        if self.feedback_type:
            self.cur_feedback = self.cur_x if self.feedback_type == 'state' else self.cur_y
            # self.cur_feedback may be attacked before implement
        else:
            self.cur_feedback = None

        return self.cur_index
