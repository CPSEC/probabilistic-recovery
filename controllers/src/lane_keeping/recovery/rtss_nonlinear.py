
# Import numpy and scipy
from numpy import pi
import numpy as np
from scipy.signal import StateSpace
# Import Rtss
from recovery.utils.formal.gaussian_distribution import GaussianDistribution
from recovery.utils.formal.reachability import ReachableSet
from recovery.utils.formal.zonotope import Zonotope
from recovery.utils.formal.strip import Strip
from recovery.utils.observers.full_state_nonlinear_from_gaussian import Estimator as fsn
from recovery.utils.observers.extended_kalman_filter import ExtendedKalmanFilter
# Import system model
class RTSSNonlinear:
	
	def __init__(self, ode, dt, W, u_min, u_max, k_reconstruction, k_max, l, a, b, euler=True, fd=None, jfx=None, jfu=None, jh=None, isolation=False, Q=None, R=None):
		'''
		Inputs
		A, B, C, D: system matrices
		W: Noise matrix
		u_min, u_max: minimum and maximum control
		k_reconstruction: maximum states that can be reconstructed
		k_max: maximum number of steps to compute the reconstruction k_reconstruction >= k_max
		'''
		assert k_reconstruction >= k_max
		# Create zonotope
		self.U = Zonotope.from_box(u_min, u_max)
		# Create reachable set
		self.k_max = k_max
		self.k_reconstruction = k_reconstruction
		self.reach = []
		# Create strip
		self.s = Strip(l, a=a, b=b)
		# Store noise cov matrix
		self.W = W
		# Create kalman filter
		self.kf = None
		self.x_cur_update = None
		self.n = len(l)
		self.m = len(u_min)
		self.isolation = isolation
		self.observer = fsn(ode, self.n, self.m, dt, W, euler, self.isolation, fd, jfx, jfu, jh, Q, R)
		self.x_0 = None
	
	
	def recovery(self, sysd, x_cur):
		reach = ReachableSet(sysd.A, sysd.B, self.U, self.W, max_step=self.k_reconstruction, c=sysd.c)
		reach.init(x_cur, self.s)
		k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=self.k_max)
		rec_u = self.U.alpha_to_control(alpha)
		return rec_u[0], k # receding horizon control
	
	def recovery_fs(self, us_checkpoint, x_checkpoint, ys_checkpoint=None):
		'''
		Assumes set_kalman_filter has been called before
		States have also been stored
		'''
		self.x_0 = GaussianDistribution(x_checkpoint, np.zeros((self.n, self.n)))
		self.x_0, sysd = self.observer.estimate(self.x_0, us_checkpoint, ys_checkpoint)
		return self.recovery(sysd, self.x_0)

	
	
	def recovery_ns(self, cur_u, y=None):
		'''
		States have also been stored
		'''
		if not self.isolation:
			self.x_0, sysd = self.observer.estimate(self.x_0, [cur_u])
		else:
			self.x_0, sysd = self.observer.estimate(self.x_0, [cur_u], [y])
		return self.recovery(sysd, self.x_0)
		
