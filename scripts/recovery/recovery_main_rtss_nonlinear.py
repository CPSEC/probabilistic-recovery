import numpy as np
from recovery.System import SystemModel
from recovery.rtss_nonlinear import RTSSNonlinear
from recovery.utils.formal.gaussian_distribution import GaussianDistribution
import copy

class RecoveryRTSSNonlinear():
	def __init__(self, dt, u_min, u_max, attacked_sensor, isolation, noise):
		system_model = SystemModel(dt, u_min, u_max)
		u_min += system_model.u0
		u_max += system_model.u0
		system_model.update_u_constraints(u_min, u_max)
		self.system_model = system_model
		# 0.00001
		# No noise 0.0025
		# Noise 0.01: 0.01
		if isolation:
			self.W = (noise/3.2 + 0.0024) * np.eye(system_model.n)
			# self.W[-1, -1] = noise/2   + 0.001
			# self.W[-2, -2] = noise/2   + 0.001
			# self.W[-3, -3] = noise/2 + 0.001
			# self.W = self.W * 2 
			self.W[-1, -1] = self.W[-2, -2] = self.W[-3, -3] = noise * 1.32 + 0.0025
			self.W[ 0,  0] = self.W[ 1,  1] = noise * 1.45 + 0.086
			self.W[2, 2] = noise / 1.55 + 0.0025 
			self.W = self.W / 1.38
			if noise > 0.003:
				self.W = self.W / 1.55
			if noise == 0.002 or noise == 0.003:
				self.W = self.W / 1.35
		else:
			self.W = (noise/3.2 + 0.0023) * np.eye(system_model.n)
			self.W[-1, -1] = self.W[-2, -2] = self.W[-3, -3] = noise * 1.31 + 0.0022
			self.W[ 0,  0] = self.W[ 1,  1] = noise * 1.32 + 0.082
			self.W[2, 2] = noise / 1.48 + 0.0022
			self.W = self.W / 2
			if noise > 0.003:
				self.W = self.W / 1.4
			if noise == 0.002 or noise == 0.003:
				self.W = self.W / 1.35
		mu = np.zeros((self.system_model.n))
		self.W = GaussianDistribution.from_standard(mu, self.W)

		l = np.array([0.5, 0.5,  -1,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  0,  0,  0])
		# l[attacked_sensor] = 1
		b = 10.2
		a = 9.8

		euler = False
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
		self.rtss = RTSSNonlinear(self.system_model.ode, self.dt, self.W, self.u_min, self.u_max, k_reconstruction=5, k_max=4,\
			l=self.l, a=self.a, b=self.b, euler=self.euler, fd=self.system_model.fd, jfx=self.system_model.jfx,\
			jfu=self.system_model.jfu, isolation=self.isolation)
	
	def init_closed_loop(self, C):
		self.x_checkpoint = []
		self.u_checkpoint = []
		self.y_checkpoint = []
		jh = lambda x, u: C
		self.C_kf = C
		Q = self.W.sigma
		R = np.eye(C.shape[0]) * 1e-2
		self.rtss = RTSSNonlinear(self.system_model.ode, self.dt, self.W, self.u_min, self.u_max, k_reconstruction=5, k_max=4,\
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
		else:
			self.u_reconf, self.k_recovery_max = self.rtss.recovery_fs(self.u_checkpoint, self.x_checkpoint)
		self.k_recovery = 0


		fM = self.u_reconf + self.system_model.u0 * 0
		fM = self.convert_input(fM)
		print("number of recovery steps: ", self.k_recovery_max)
		return fM, self.k_recovery_max
	
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

		fM = copy.deepcopy(self.u_reconf) + self.system_model.u0 * 0
		# print(self.u_reconf)
		fM = self.convert_input(fM)
		self.k_recovery += 1
		return fM, self.k_recovery_max

	def convert_input(self, fM):
		fM[2] = -fM[2]
		fM[3] = -fM[3]
		return fM



		
	

	
	
