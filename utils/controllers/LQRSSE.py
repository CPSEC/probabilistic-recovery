from scipy.linalg import solve_continuous_are, inv
import numpy as np
from .controller_base import Controller


class LQRSSE(Controller):
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.update_gain(A, B, Q, R)
        self.ref = 0
        self.control_lo = None
        self.control_up = None

    def update_gain(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        P = solve_continuous_are(A, B, Q, R)
        self.K = inv(R) @ (B.T @ P)

    def update(self, feedback_value: np.ndarray, current_time=None) -> np.ndarray:
        # cin = -self.K @ (feedback_value - self.ref)
        # if self.control_lo and self.control_up:
        #     for i in range(len(cin)):
        #         cin[i] = np.clip(cin[i], self.control_lo[i], self.control_up[i])
        cin = -self.K @ (feedback_value - self.ref * self.Nx )
        cin = cin + feedback_value * self.Nu
        if self.control_lo and self.control_up:
            for i in range(len(cin)):
                cin[i] = np.clip(cin[i], self.control_lo[i], self.control_up[i])
        return cin

    def set_control_limit(self, control_lo: np.ndarray, control_up: np.ndarray):
        self.control_lo = control_lo
        self.control_up = control_up

    def set_reference(self, ref: np.ndarray):
        self.ref = ref

    def compute_Nbar(self, A:np.ndarray, B:np.ndarray, C:np.ndarray, D:np.ndarray):
        s = A.shape[0]
        Z = np.zeros([1, s])
        Z = np.concatenate((Z, np.array([[1]])), axis=1)
        con = np.concatenate((A, B), axis=1)
        con2 = np.concatenate((C, D), axis=1)
        temp = np.concatenate((con, con2), axis=0)
        N = inv(temp) @ np.transpose(Z)
        self.Nx = N[0:s]
        self.Nu = N[s]
        self.Nbar=self.Nu + self.K*self.Nx

