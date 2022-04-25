import numpy as np
from control.matlab import ss, c2d


class Simulator:
    def __init__(self, name, model_type='linear'):
        self.name = name
        self.model_type = model_type
        self.dt = 0
        self.cur_x = None
        self.cur_y = None
        self.cur_u = None
        self.cur_index = 0
        self.inputs = []
        self.outputs = []
        self.states = []

    @classmethod
    def linear(cls, name, A, B, C=None, D=None, Ts=None):
        sim = cls(name, model_type='linear')
        sim.dt = Ts if Ts else 0
        if C is None:
            C = np.eye(len(A))
        if D is None:
            D = 0
        sim.sysc = ss(A, B, C, D)
        if Ts is not None:
            sim.sysd = c2d(sim.sysc, Ts)
        return sim

    def set_init_state(self, x):
        self.init_x = np.array(x)
        self.init_y = self.sysd.C @ self.init_x
        self.outputs.append(self.init_y)
        self.states.append(self.init_x)

    def set_feedback(self, feedback):
        """
        'state', 'output', None
        """
        self.feedback = feedback

    def set_controller(self, controller):
        """
        please implement update method to get control input
        """
        self.controller = controller

    def evolve(self, u=None):
        if self.feedback:
            feeddata = self.cur_x if self.feedback == 'state' else self.cur_y
            self.cur_u = self.controller.update(feeddata, self.dt*self.cur_index)
        else:
            self.cur_u = u

        self.cur_index += 1

