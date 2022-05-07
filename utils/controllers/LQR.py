from scipy.linalg import solve_continuous_are, inv
import numpy as np

class LQR:
    def __init__(self, A, B, Q, R):
        self.update_gain(A, B, Q, R)
        self.ref = 0
        self.control_limit = None

    def update_gain(self, A, B, Q, R):
        P = solve_continuous_are(A, B, Q, R)
        self.K = inv(R) @ (B.T @ P)

    def update(self, feedback_value, current_time=None):
        cin = -self.K @ (feedback_value - self.ref)
        if self.control_limit:
            for i in range(len(cin)):
                cin[i] = np.clip(cin[i], self.control_limit['lo'][i], self.control_limit['up'][i])
        return cin

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up

    def set_reference(self, ref):
        self.ref = ref