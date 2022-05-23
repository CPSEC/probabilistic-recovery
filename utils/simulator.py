import numpy as np
from scipy.signal import StateSpace
from scipy.integrate import solve_ivp


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
        # values under self.cur_index
        self.cur_x = None
        self.cur_y = None
        self.cur_u = None
        self.cur_feedback = None
        self.cur_ref = None
        self.cur_index = 0
        self.m_noise = None
        self.p_noise = None

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

    def linear(self, A, B, C=None, D=None):
        self.model_type = 'linear'
        if C is None:
            C = np.eye(len(A))
        if D is None:
            D = np.zeros((len(C), len(B[0])))
        self.sysc = StateSpace(A, B, C, D)
        self.sysd = self.sysc.to_discrete(self.dt)
        self.n = self.sysc.A.shape[1]
        self.m = self.sysc.B.shape[1]
        self.p = self.sysc.C.shape[0]
        self.C = self.sysc.C
        self.D = self.sysc.D

        def fprime(t, x, u):
            return self.sysc.A @ x + self.sysc.B @ u

        self.ode = fprime

    def nonlinear(self, ode, n, m, p, C=None, D=None):
        self.model_type = 'nonlinear'
        self.C = np.eye(n) if C is None else np.array(C)
        self.D = np.zeros((p, m)) if D is None else np.array(D)
        self.n = n
        self.m = m
        self.p = p
        self.ode = ode

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
              np.array([sigma_1, sigma_2, ..., sigma_{n/m}])      scale for 'white'
        """
        if 'process' in noise:
            if noise['process']['type'] == 'white':
                scale = noise['process']['param'].reshape((1, -1))
                self.p_noise = np.random.normal(0, 1, (self.max_index + 2, self.n)) * scale
        if 'measurement' in noise:
            if noise['measurement']['type'] == 'white':
                scale = noise['measurement']['param'].reshape((-1, 1))
                self.m_noise = np.random.normal(0, 1, (self.max_index + 2, self.p)) * scale

    def set_init_state(self, x):
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

    def evolve(self, u=None):
        # record data
        self.feedbacks[self.cur_index] = self.cur_feedback
        self.refs[self.cur_index] = self.cur_ref

        # compute control input
        if self.feedback_type:
            self.cur_u = self.controller.update(self.cur_ref, self.cur_feedback, self.dt * self.cur_index)
        else:
            self.cur_u = u
        assert self.cur_u.shape == (self.m,)
        self.inputs[self.cur_index] = self.cur_u

        # implement control input
        ts = (self.cur_index * self.dt, (self.cur_index + 1) * self.dt)
        res = solve_ivp(self.ode, ts, self.cur_x, args=(self.cur_u,))
        self.cur_index += 1
        self.cur_x = res.y[:, -1]
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