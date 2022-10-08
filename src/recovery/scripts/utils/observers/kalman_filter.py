import numpy as np
from scipy.linalg import inv

class KalmanFilter:
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray, C: np.ndarray, D: [np.ndarray, None], Q: np.ndarray, R: np.ndarray):
        self.A = Ad
        self.B = Bd
        self.C = C
        self.D = D if D is None else np.zeros((self.C.shape[0], self.B.shape[1]))
        self.Q = Q
        self.R = R

    def predict(self, x: np.ndarray, P: np.ndarray, u: np.ndarray):
        x_predict = self.A @ x + self.B @ u
        P_predict = self.A @ P @ self.A.T + self.Q
        return x_predict, P_predict

    def update(self, x_predict, P_predict, y: np.ndarray):
        K = P_predict @ self.C.T @ inv(self.C @ P_predict @ self.C.T + self.R)
        x_update = x_predict + K @ (y-self.C @ x_predict)
        P_update = P_predict - K @ self.C @ P_predict
        return x_update, P_update

    def one_step(self, x: np.ndarray, P: np.ndarray, u: np.ndarray, y: np.ndarray):
        x_predict, P_predict = self.predict(x, P, u)
        x_update, P_update = self.update(x_predict, P_predict, y)
        return x_update, P_update

    def multi_steps(self, x: np.ndarray, P: np.ndarray, us: np.ndarray, ys: np.ndarray):
        """
        us: shape (length, m)
        ys: shape (length, p)
        """
        assert len(us) == len(ys)
        length = len(us)
        x_res = np.empty((length+1, *x.shape))
        P_res = np.empty((length+1, *P.shape))
        x_res[0] = x
        P_res[0] = P
        for i in range(length):
            x_res[i+1], P_res[i+1] = self.one_step(x_res[i], P_res[i], us[i], ys[i])
        return x_res, P_res

