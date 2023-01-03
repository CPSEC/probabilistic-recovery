import numpy as np
from recovery.System import SystemModel
from recovery.rtss_nonlinear import RTSSNonlinear
from recovery.utils.formal.gaussian_distribution import GaussianDistribution

class RecoveryRTSSNonlinear():
	def __init__(self, ref_after_attack, dt, u_min, u_max, isolation):
		system_model = SystemModel(dt, u_min, u_max)
		self.system_model = system_model
		# 0.00001
		# No noise 0.0025
		# Noise 0.01: 0.01
		if isolation:
			self.W = 0.005 * np.eye(system_model.n) #*20
		else:
			self.W = 0.000001 * np.eye(system_model.n) #*20
		# self.W = self.W *15
		mu = np.zeros((self.system_model.n))
		self.W = GaussianDistribution.from_standard(mu, self.W)

		l = np.array([1, 0])
		# l[attacked_sensor] = 1
		a = l @ (ref_after_attack - self.system_model.x0*0) - 0.2
		b = l @ (ref_after_attack - self.system_model.x0*0) + 0.2

		euler = True
		self.dt = dt
		self.u_min = u_min
		self.u_max = u_max
		self.l = l
		self.a = a
		self.b = b
		self.euler = euler
		self.isolation = isolation
		self.u_reconf = []
		self.k_recovery_max = -1

	###################
	## Auxiliar functions
	###################
	def init_open_loop(self):
		self.x_checkpoint = []
		self.u_checkpoint = []
		print("init open loop")
		self.rtss = RTSSNonlinear(self.system_model.ode, self.dt, self.W, self.u_min, self.u_max, k_reconstruction=11, k_max=10,\
			l=self.l, a=self.a, b=self.b, euler=self.euler, fd=self.system_model.fd, jfx=self.system_model.jfx,\
			jfu=self.system_model.jfu, isolation=self.isolation)
	
	def init_closed_loop(self, C):
		self.x_checkpoint = []
		self.u_checkpoint = []
		self.y_checkpoint = []
		jh = lambda x, u: C
		self.C_kf = C
		Q = self.W.sigma
		R = np.eye(C.shape[0]) * 1e-4
		self.rtss = RTSSNonlinear(self.system_model.ode, self.dt, self.W, self.u_min, self.u_max, k_reconstruction=11, k_max=10,\
			l=self.l, a=self.a, b=self.b, euler=self.euler, fd=self.system_model.fd, jfx=self.system_model.jfx,\
			jfu=self.system_model.jfu, jh=jh, isolation=self.isolation, Q=Q, R=R)
		

	def checkpoint_state(self, state):
		self.x_checkpoint = state - self.system_model.x0*0

	def checkpoint(self, x, u):
		u = u.flatten()
		# print(u)
		# print(self.system_model.u0)
		du = u - self.system_model.u0*0
		if self.isolation:
			dy = self.C_kf @ (x - self.system_model.x0*0)
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
		self.k_recovery_max = 0
		if self.isolation:
			self.u_reconf, self.k_recovery_max = self.rtss.recovery_fs(self.u_checkpoint, self.x_checkpoint, self.y_checkpoint)
			str_cmd = self.u_reconf 
		else:
			self.u_reconf, self.k_recovery_max = self.rtss.recovery_fs(self.u_checkpoint, self.x_checkpoint)
			str_cmd = self.u_reconf
		self.k_recovery = 0

		str_cmd = str_cmd + self.system_model.u0 * 0
		print("number of recovery steps: ", self.k_recovery_max)
		return str_cmd, self.k_recovery_max
	
	########################
	# Recovery last iterations
	########################
	def update_recovery_ni(self, state, u):
		
		u = u.flatten()
		dx = state - self.system_model.x0*0
		du = u - self.system_model.u0*0

		if self.isolation:
			self.u_reconf, self.k_recovery_max = self.rtss.recovery_ns(du, self.C_kf @ dx)
		else:
			self.u_reconf, self.k_recovery_max = self.rtss.recovery_ns(du)
		str_cmd = self.u_reconf 
		str_cmd = str_cmd + self.system_model.u0 * 0

		self.k_recovery += 1
		return str_cmd, self.k_recovery_max





		
	

	
	
