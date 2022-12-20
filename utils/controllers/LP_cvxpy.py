from typing import Union

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

from utils.controllers.controller_base import Controller


class LP(Controller):
    def __init__(self, param_dict: dict):
        assert 'Bd' in param_dict
        self.update_params(**param_dict)

    def update_params(self, **kwargs):
        """
        parameter template:
        lp_settings = {
            'Ad': ,  'Bd': ,
            'N': ,
            'ddl': , 'target_lo': , 'target_up': ,
            'safe_lo': , 'safe_up': ,
            'control_lo': , 'control_up': ,
            'ref':
        }
        """
        if 'Ad' in kwargs and 'Bd' in kwargs:
            self.update_model(kwargs['Ad'], kwargs['Bd'])
        if 'N' in kwargs:
            self.update_horizon(kwargs['N'])
        if 'ddl' in kwargs:
            self.update_ddl(kwargs['ddl'])
        if 'target_lo' in kwargs and 'target_up' in kwargs:
            self.update_target_set(kwargs['target_lo'], kwargs['target_up'])
        if 'safe_lo' in kwargs and 'safe_up' in kwargs:
            self.update_safe_set(kwargs['safe_lo'], kwargs['safe_up'])
        if 'control_lo' in kwargs and 'control_up' in kwargs:
            self.set_control_limit(kwargs['control_lo'], kwargs['control_up'])
        if 'ref' in kwargs:
            self.set_reference(kwargs['ref'])
        if 'solver' in kwargs:
            self.set_solver(kwargs['solver'])

        if 'c_nonlinear' in kwargs:
            self.update_model_residual(kwargs['c_nonlinear'])
        else:
            self.update_model_residual(None)

    def ready_to_formulate(self):
        required = ['Ad', 'Bd', 'N', 'xr', 'x0']
        ddl_required = ['xtmin', 'xtmax']
        short = []
        for key in required:
            if not hasattr(self, key):
                short.append(key)
        if hasattr(self, 'ddl'):
            for key in ddl_required:
                if not hasattr(self, key):
                    short.append(key)
        # default values
        if not hasattr(self, 'xmin'):
            self.update_safe_set(np.array([-np.inf] * self.nx), np.array([np.inf] * self.nx))
        if not hasattr(self, 'umin'):
            self.set_control_limit(np.array([-np.inf] * self.nu), np.array([np.inf] * self.nu))
        if len(short) == 0:
            return True
        else:
            print(short, 'are required but not provided.')
            return False

    def update_model(self, Ad: np.ndarray, Bd: np.ndarray):
        self.Ad = Ad
        self.Bd = Bd
        self.nx, self.nu = Bd.shape

    def update_model_residual(self, cd: np.ndarray):
        if cd is not None:
            self.cd = cd
        else:
            self.cd = np.zeros(self.nx)

    def update_horizon(self, N: int):
        self.N = N

    def update_ddl(self, ddl: Union[int, None]):
        if ddl is None:
            if hasattr(self, 'ddl'):
                delattr(self, 'ddl')
        else:
            self.ddl = ddl
        assert not hasattr(self, 'ddl') or self.ddl <= self.N

    def update_target_set(self, target_lo: np.ndarray, target_up: np.ndarray):
        self.xtmin = target_lo
        self.xtmax = target_up

    def update_safe_set(self, safe_lo: np.ndarray, safe_up: np.ndarray):
        self.xmin = safe_lo
        self.xmax = safe_up

    def formulate(self):
        """
           x = | x0, x1, ..., xN, u0, u1, ..., u{N-1} |
           Aeq                                     leq (1-D)
            | -I                              |     | -x0 |
            | Ad -I              Bd           |     |     |
            |    Ad -I              Bd        |     |     |
            |        ...             ...      |     |     |
            |           Ad -I            Bd   |     |     |
        """
        if not self.ready_to_formulate():
            raise ValueError('Not ready to formulate')
        Ax = np.kron(np.eye(self.N + 1), -np.eye(self.nx)) + np.kron(np.eye(self.N + 1, k=-1), self.Ad)
        Bu = np.kron(np.vstack([np.zeros((1, self.N)), np.eye(self.N)]), self.Bd)
        self.Aeq = np.hstack([Ax, Bu])
        self.Aineq = np.eye((self.N + 1) * self.nx + self.N * self.nu)

        repeated_cd = np.array(self.cd.tolist() * self.N)
        self.leq = np.hstack([-self.x0, -1*repeated_cd]) # self.leq = np.hstack([-self.x0, np.zeros(self.N * self.nx)])
        if hasattr(self, 'ddl') and self.ddl <= self.N:
            self.lineq = np.hstack(
                [np.kron(np.ones(self.ddl), self.xmin), np.kron(np.ones(self.N - self.ddl + 1), self.xtmin),
                 np.kron(np.ones(self.N), self.umin)])
            self.uineq = np.hstack(
                [np.kron(np.ones(self.ddl), self.xmax), np.kron(np.ones(self.N - self.ddl + 1), self.xtmax),
                 np.kron(np.ones(self.N), self.umax)])
        else:
            self.lineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmin), np.kron(np.ones(self.N), self.umin)])
            self.uineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmax), np.kron(np.ones(self.N), self.umax)])

        self.x = cp.Variable((self.N + 1) * self.nx + self.N * self.nu)
        self.prob = cp.Problem(cp.Minimize(0),
                               [self.Aineq @ self.x <= self.uineq,
                                self.lineq <= self.Aineq @ self.x,
                                self.Aeq @ self.x == self.leq])

    def formulate_only_x0(self):
        self.leq[:self.nx] = -self.x0
        # self.prob = cp.Problem(self.prob.objective,
        #                        [self.Aineq @ self.x <= self.uineq,
        #                         self.lineq <= self.Aineq @ self.x,
        #                         self.Aeq @ self.x == self.leq])

    def update(self, feedback_value: np.ndarray, current_time=None):
        self.x0 = feedback_value
        if not hasattr(self, 'prob'):
            self.formulate()
        else:
            self.formulate_only_x0()
        self.prob.solve(solver=self.solver, warm_start=True)

        if self.prob.status != cp.OPTIMAL:
            print('did not get an optimal solution!')
            raise ValueError('did not solve the problem!')
        # Apply first control input to the plant
        # res.x = [ x0, x1, ... xN, u0, u1, u{N-1} ]
        ctrl = self.x.value[-self.N * self.nu:-(self.N - 1) * self.nu]
        return ctrl

    def get_full_ctrl(self):
        """
        Only call after self.update to get full control input sequence  u0,...,u{N-1}
        """
        return self.x.value[-self.N * self.nu:].reshape((-1, self.nu))

    def get_last_x(self):
        """
        Only call after self.update to get last system state xN for debug
        """
        return self.x.value[self.N * self.nx:(self.N+1) * self.nx]

    def set_control_limit(self, control_lo: np.ndarray, control_up: np.ndarray):
        self.umin = control_lo
        self.umax = control_up

    def set_reference(self, ref: np.ndarray):
        self.xr = ref

    def set_solver(self, solver):
        self.solver = solver
