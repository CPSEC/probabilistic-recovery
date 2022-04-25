import numpy as np
from control.matlab import ss, c2d, lsim


class Simulator:
    def __init__(self, name, Ts, max_index):
        self.name = name
        self.model_type = None
        self.dt = Ts
        self.sysc = None
        self.sysd = None
        self.max_index = max_index
        # values under self.cur_index
        self.cur_x = None
        self.cur_y = None
        self.cur_u = None
        self.cur_feedback = None
        self.cur_ref = None
        self.cur_index = 0
        # backup all data
        self.inputs = np.empty(max_index + 2, dtype=object)
        self.outputs = np.empty(max_index + 2, dtype=object)
        self.states = np.empty(max_index + 2, dtype=object)
        self.feedbacks = np.empty(max_index + 2, dtype=object)
        self.refs = np.empty(max_index + 2, dtype=object)  # reference value

    def linear(self, A, B, C=None, D=None):
        self.model_type = 'linear'
        if C is None:
            C = np.eye(len(A))
        if D is None:
            D = 0
        self.sysc = ss(A, B, C, D)
        self.sysd = c2d(self.sysc, self.dt)

    def sim_init(self, settings):
        self.set_feedback_type(settings['feedback_type'])
        self.set_init_state(settings['init_state'])
        self.set_controller(settings['controller'])

    def set_init_state(self, x):
        self.cur_x = x
        self.cur_y = self.sysd.C @ self.cur_x
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
        self.inputs[self.cur_index] = self.cur_u

        # implement control input
        self.cur_index += 1
        if self.model_type == 'linear':
            self.cur_x = self.sysd.A @ self.cur_x + self.sysd.B @ self.cur_u
            self.cur_y = self.sysd.C @ self.cur_x + self.sysd.D @ self.cur_u
            self.states[self.cur_index] = self.cur_x
            self.outputs[self.cur_index] = self.cur_y

        # prepare feedback
        if self.feedback_type:
            self.cur_feedback = self.cur_x if self.feedback_type == 'state' else self.cur_y
            # self.cur_feedback may be attacked before implement
        else:
            self.cur_feedback = None

        return self.cur_index
