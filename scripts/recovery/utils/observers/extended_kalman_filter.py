import numpy as np
from scipy.linalg import inv

class ExtendedKalmanFilter:
	def __init__(self, fd, jf, jh, Q: np.ndarray, R: np.ndarray):
		self.A = []
		self.C = []
		self.Q = Q
		self.R = R
		self.fd = fd
		self.jf = jf
		self.jh = jh
		self.n = len(Q)

	def predict(self, x: np.ndarray, P: np.ndarray, u: np.ndarray):
		x_predict = self.fd(x, u)
		self.A = self.jf(x, u)
		# print(self.A)
		self.C = self.jh(x_predict, u)
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
		return x_update.flatten(), P_update

