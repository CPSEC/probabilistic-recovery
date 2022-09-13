from typing import Union

import numpy as np
from scipy import sparse
import osqp

from utils.controllers.controller_base import Controller


class MPC(Controller):
    def __init__(self, param_dict: dict):
        assert 'Bd' in param_dict
        self.update_params(**param_dict)

    def update_params(self, **kwargs):
        """
        parameter template:
        mpc_settings = {
            'Ad': ,  'Bd': ,
            'Q': , 'QN':, 'R':,
            'N': ,
            'ddl': , 'target_lo': , 'target_up': ,
            'safe_lo': , 'safe_up': ,
            'control_lo': , 'control_up': ,
            'ref':
        }
        """
        if 'Ad' in kwargs and 'Bd' in kwargs:
            self.update_model(kwargs['Ad'], kwargs['Bd'])
        if 'Q' in kwargs and 'QN' in kwargs and 'R' in kwargs:
            self.update_object(kwargs['Q'], kwargs['QN'], kwargs['R'])
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

    def ready_to_formulate(self):
        required = ['Ad', 'Bd', 'Q', 'QN', 'R', 'N', 'xr', 'x0']
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
            self.update_safe_set(np.array([-np.inf]*self.nx), np.array([np.inf]*self.nx))
        if not hasattr(self, 'umin'):
            self.set_control_limit(np.array([-np.inf]*self.nu), np.array([np.inf]*self.nu))
        if len(short) == 0:
            return True
        else:
            print(short, 'are required but not provided.')
            return False

    def update_model(self, Ad: np.ndarray, Bd: np.ndarray):
        self.Ad = Ad
        self.Bd = Bd
        self.nx, self.nu = Bd.shape

    def update_object(self, Q: np.ndarray, QN: np.ndarray, R: np.ndarray):
        self.Q = Q
        self.QN = QN
        self.R = R

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
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN, sparse.kron(sparse.eye(self.N), self.R)], format='csc')
        q = np.hstack([np.kron(np.ones(self.N), -self.Q @ self.xr), -self.QN @ self.xr, np.zeros(self.N*self.nu)])
        Ax = sparse.kron(sparse.eye(self.N+1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N+1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        A = sparse.vstack([Aeq, Aineq], format='csc')

        leq = np.hstack([-self.x0, np.zeros(self.N * self.nx)])
        ueq = leq
        if hasattr(self, 'ddl') and self.ddl < self.N:
            lineq = np.hstack([np.kron(np.ones(self.ddl), self.xmin), np.kron(np.ones(self.N-self.ddl+1), self.xtmin), np.kron(np.ones(self.N), self.umin)])
            uineq = np.hstack([np.kron(np.ones(self.ddl), self.xmax), np.kron(np.ones(self.N-self.ddl+1), self.xtmax), np.kron(np.ones(self.N), self.umax)])
        else:
            lineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmin), np.kron(np.ones(self.N), self.umin)])
            uineq = np.hstack([np.kron(np.ones(self.N + 1), self.xmax), np.kron(np.ones(self.N), self.umax)])
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])

        self.prob = osqp.OSQP()
        self.prob.setup(P, q, A, self.l, self.u, warm_start=True)

    def formulate_only_x0(self):
        self.l[:self.nx] = -self.x0
        self.u[:self.nx] = -self.x0
        self.prob.update(l=self.l, u=self.u)

    def update(self, feedback_value: np.ndarray, current_time=None):
        self.x0 = feedback_value
        if not hasattr(self, 'prob'):
            self.formulate()
        else:
            self.formulate_only_x0()
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        # Apply first control input to the plant
        # res.x = [ x0, x1, ... xN, u0, u1, u{N-1} ]
        ctrl = res.x[-self.N * self.nu:-(self.N - 1) * self.nu].reshape((-1, self.nu))
        ctrl_all = res.x[-self.N * self.nu:].reshape((-1, self.nu))
        return ctrl, ctrl_all

    def set_control_limit(self, control_lo: np.ndarray, control_up: np.ndarray):
        self.umin = control_lo
        self.umax = control_up

    def set_reference(self, ref: np.ndarray):
        self.xr = ref