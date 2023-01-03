import numpy as np
from recovery.System import SystemModel
from recovery.rtss import RTSS
from recovery.utils.formal.gaussian_distribution import GaussianDistribution

class RecoveryRTSS():
	def __init__(self, ref_after_attack, dt, u_min, u_max, isolation):
		system_model = SystemModel(dt, u_min, u_max)
		self.system_model = system_model
		# 0.00001
		# No noise 0.0025
		# Noise 0.01: 0.01
		if isolation:
			self.W = 0.001 * np.eye(system_model.n) #*20
		else:
			self.W = 0.0001 * np.eye(system_model.n) #*20
		# self.W = self.W *15
		mu = np.zeros((self.system_model.n))
		self.W = GaussianDistribution.from_standard(mu, self.W)

		l = np.array([1, 0])
		# l[attacked_sensor] = 1
		a = l @ (ref_after_attack - self.system_model.x0) - 0.2
		b = l @ (ref_after_attack - self.system_model.x0) + 0.2


		self.rtss = RTSS(system_model.Ad, system_model.Bd, system_model.Cd, system_model.Dd, self.W, u_min, u_max, k_reconstruction=61, k_max=60, l=l, a=a, b=b )
		self.isolation = isolation
		
		self.k_recovery = -1
		self.u_reconf = []
		self.k_max = -1
	
	###################
	## Auxiliar functions
	###################
	def init_open_loop(self):
		self.x_checkpoint = []
		self.u_checkpoint = []
	
	def init_closed_loop(self, C):
		self.x_checkpoint = []
		self.u_checkpoint = []
		self.y_checkpoint = []
		self.C_kf = C
		R = np.eye(C.shape[0]) *0
		self.rtss.set_kalman_filter(C, self.W.sigma, R)
		
	
	def checkpoint_state(self, state):
		self.x_checkpoint = state - self.system_model.x0

	def checkpoint(self, x, u):
		u = u.flatten()
		print(u)
		print(self.system_model.u0)
		du = u - self.system_model.u0
		if self.isolation:
			dy = self.C_kf @ (x - self.system_model.x0)
			self.checkpoint_closed_loop(dy, du)
		else:
			self.checkpoint_input_open_loop(du)
	
	# Checkpoint the input for the open loop strategy
	def checkpoint_input_open_loop(self, du_0):
		self.u_checkpoint.append(du_0)

	# Checkpoint the input and measurement for the closed loop strategy
	def checkpoint_closed_loop(self, dy, du):
		self.y_checkpoint.append(dy)
		self.u_checkpoint.append(du)
		pass

	########################
	# Recovery first iteration
	########################
	def update_recovery_fi(self):
		self.k_max = 0
		if self.isolation:
			self.u_reconf, self.k_max = self.rtss.recovery_isolation_fs(self.u_checkpoint, self.y_checkpoint, self.x_checkpoint)
			str_cmd = self.u_reconf[0] # receding horizon control
		else:
			self.u_reconf, self.k_max = self.rtss.recovery_no_isolation(self.u_checkpoint, self.x_checkpoint)
			str_cmd = self.u_reconf[0] # Take the first u
		self.k_recovery = 0

		str_cmd = str_cmd + self.system_model.u0
		print("number of recovery steps: ", self.k_max)
		return str_cmd, self.k_max
	
	########################
	# Recovery final iterations
	########################
	def update_recovery_ni(self, state, u):
		
		
		u = u.flatten()

		dx = state - self.system_model.x0
		du = u - self.system_model.u0
		self.k_recovery += 1
		if self.isolation:
			str_cmd, self.k_max = self.rtss.recovery_isolation_ns(dx, du)
		else:
			str_cmd = self.u_reconf[self.k_recovery] # look up vector
			self.k_max -= 1
		# print(str_cmd)

		str_cmd = str_cmd + self.system_model.u0
		return str_cmd, self.k_max





		
	

	
	
