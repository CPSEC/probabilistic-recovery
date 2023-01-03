import numpy as np
from recovery.System import SystemModel
from recovery.utils.observers.full_state_bound import Estimator


class RecoveryVirtualSensor():
	def __init__(self, ref_after_attack, dt, u_min, u_max):
		self.u_min = u_min
		self.u_max = u_max
		system_model = SystemModel(dt, u_min, u_max)
		self.system_model = system_model

		self.estimator = Estimator(system_model.Ad, system_model.Bd, max_k = 150, epsilon= 1e-3)

		target_lo = np.array([ref_after_attack[0] - 0.2, -2*np.pi])
		target_lo = target_lo - system_model.x0
		self.target_lo = target_lo

		target_up = np.array([ref_after_attack[0] + 0.2, 2*np.pi]) 
		target_up = target_up - system_model.x0
		self.target_up = target_up

	
	def init_open_loop(self):
		self.x_checkpoint = []
		self.u_checkpoint = []
	
	def init_closed_loop(self, C):
		pass
		

	def checkpoint_state(self, state):
		x = state
		self.x_checkpoint = x - self.system_model.x0

	def checkpoint(self, x, u):
		u = u.flatten()
		du = u - self.system_model.u0
		self.checkpoint_input_open_loop(du)
	
	# Checkpoint the input for the open loop strategy
	def checkpoint_input_open_loop(self, du_0):
		self.u_checkpoint.append(du_0)

	# Checkpoint the input and measurement for the closed loop strategy
	def checkpoint_closed_loop(self, dy, du):
		pass

	# Auxiliar function to call the recovery for the first time
	def update_recovery_fi(self):
		x_cur_lo, x_cur_up, x_cur = self.estimator.estimate(self.x_checkpoint, self.u_checkpoint)
		x_cur = x_cur + self.system_model.x0
		y = self.system_model.Cd @ x_cur
		return y.flatten()

	# Computes the estimated state after first iteration
	def update_recovery_ni(self, x, u):
		# We are linearizing the model. So we define dx = x - x0, x0 the linearization point
		x = x - self.system_model.x0
		u = u.flatten() - self.system_model.u0

		x_cur_lo, x_cur_up, x_cur = self.estimator.estimate(x, [u])

		# We are linearizing the model. So we define x = dx + x0, x0 the linearization point
		x_cur = x_cur + self.system_model.x0
		y = self.system_model.Cd @ x_cur
		return y.flatten()
	
	def in_set(self, x):
		res = True
		# We are linearizing the model. So we define dx = x - x0, x0 the linearization point
		x = x - self.system_model.x0
		for i in range(len(x)):
			if self.target_lo[i] > x[i] or self.target_up[i] < x[i]:
				res = False
				break
		return res



