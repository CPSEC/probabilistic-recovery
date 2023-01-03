import numpy as np
from recovery.System import SystemModel
from recovery.utils.controllers.MPC import MPC
from recovery.utils.observers.full_state_bound import Estimator

class RecoveryEmsoft():
	def __init__(self, ref_after_attack, dt, u_min, u_max):
		self.u_min = u_min
		self.u_max = u_max
		system_model = SystemModel(dt, u_min, u_max)
		self.system_model = system_model
		self.u_reconf = []

		self.estimator = Estimator(system_model.Ad, system_model.Bd, max_k = 150, epsilon= 1e-3)


		# TODO: define the matrices correctly
						#  y, \theta
		self.Q  = np.diag([1, 0])
		# self.Q[attacked_sensor, attacked_sensor] = 1
						#  y, \theta
		self.QN = np.diag([1, 0])
		# self.QN[attacked_sensor, attacked_sensor] = 1

		self.R  = np.eye(system_model.m)/1000
		
						#   y, \theta
		safe_lo = np.array([-0.6, -np.pi])
		safe_lo = safe_lo - system_model.x0

						#   y, \theta
		safe_up = np.array([0.5, np.pi])
		safe_up = safe_up - system_model.x0


		self.safe_lo = safe_lo
		self.safe_up = safe_up

						#     y, \theta
		target_lo = np.array([ref_after_attack[0] - 0.2, -np.pi])
		target_lo = target_lo - system_model.x0
		self.target_lo = target_lo
		
						#     y, \theta
		target_up = np.array([ref_after_attack[0] + 0.2, np.pi])
		target_up = target_up - system_model.x0
		self.target_up = target_up

		ref = ref_after_attack - system_model.x0
		self.ref = ref

	def init_open_loop(self):
		self.x_checkpoint = []
		self.u_checkpoint = []
	
	def init_closed_loop(self, C):
		pass
		

	def checkpoint_state(self, state):
		self.x_checkpoint = state - self.system_model.x0

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
		x_cur_lo, x_cur_up, x_curr = self.estimator.estimate(self.x_checkpoint, self.u_checkpoint)
		control = self.u_checkpoint[-1]
		k = self.estimator.get_deadline(x_curr, self.safe_lo, self.safe_up, control, 100) + 1
		mpc_settings = {
			'Ad': self.system_model.Ad, 'Bd': self.system_model.Bd,
			'Q': self.Q, 'QN': self.QN, 'R':self.R,
			'N': k+3,
			'ddl': k, 'target_lo':self.target_lo, 'target_up':self.target_up,
			'safe_lo':self.safe_lo, 'safe_up':self.safe_up,
			'control_lo': self.u_min, 'control_up': self.u_max,
			'ref':self.ref
		}
		self.mpc = MPC(mpc_settings)

		self.k_max = k + 3
		_, rec_u = self.mpc.update(feedback_value= x_curr)
		fM = rec_u[0] # Pick the first u
		self.k_recovery = 0
		
		self.u_reconf = rec_u
		fM = fM + self.system_model.u0
		
		print("number of recovery steps: ", self.k_max)
		return fM, self.k_max
	
	# Auxiliar function to call the recovery 
	def update_recovery_ni(self, state, u):
		
		fM = self.u_reconf[self.k_recovery] # look up vector
		self.k_max -= 1
		self.k_recovery += 1
		# print(fM)


		fM = fM + self.system_model.u0
		return fM, self.k_max




		
	

	
	
