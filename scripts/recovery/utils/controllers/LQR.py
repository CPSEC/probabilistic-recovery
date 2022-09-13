from scipy.linalg import solve_continuous_are, inv
import numpy as np
from .controller_base import Controller


class LQR(Controller):
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.update_gain(A, B, Q, R)
        self.ref = 0
        self.control_lo = None
        self.control_up = None

    def update_gain(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        P = solve_continuous_are(A, B, Q, R)
        self.K = inv(R) @ (B.T @ P)

    def update(self, feedback_value: np.ndarray, current_time=None) -> np.ndarray:
        cin = -self.K @ (feedback_value - self.ref)
        if self.control_lo and self.control_up:
            for i in range(len(cin)):
                cin[i] = np.clip(cin[i], self.control_lo, self.control_up)
        return cin

    def set_control_limit(self, control_lo: np.ndarray, control_up: np.ndarray):
        self.control_lo = control_lo
        self.control_up = control_up

    def set_reference(self, ref: np.ndarray):
        self.ref = ref
