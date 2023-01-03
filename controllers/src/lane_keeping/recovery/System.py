import numpy as np
from scipy.signal import StateSpace

class SystemModel():
	def __init__(self, dt, u_min, u_max):
		self.L = 0.32
		self.lr = 0.13
		# self.lf = 0.17
		self.u0 = [0.4, 0]
		self.x0 = [0, 0, 0]
		x_linear = self.x0
		u_linear = self.u0
		self.u0 = np.array(self.u0[1])
		self.x0 = np.array(self.x0[1:])
		self.dt = dt
		self.model = "cg"
		if self.model == "rear":
			self.fd = lambda x,u: np.array( [[x[0] + u_linear[0] * np.sin(x[1]) * dt],\
				[x[1] + dt * u_linear[0] * np.tan(u[0]) / self.L]] )

			self.jfx = lambda x, u: np.array( [[1, dt * u_linear[0] * np.cos(x[1])],\
				[0, 1]] )
				
			self.jfu = lambda x, u: np.array([[0],\
				[dt * u_linear[0] * (np.tan(u[0])**2 + 1) / self.L]])

			self.ode = lambda t, x, u: np.array([[u_linear[0] * np.sin(x[1])], [u_linear[0] * np.tan(u[0]) / self.L]]).flatten()
		else:
			self.fd = lambda x, u: np.array( [[x[0] + dt * u_linear[0] * np.sin( x[1] + np.arctan( self.lr * np.tan(u[0]) / self.L ))], \
				[x[1] + dt * u_linear[0] * np.tan(u[0]) * np.cos( np.arctan(self.lr * np.tan(u[0]) / self.L) )]] )
			
			# TODO: complete this
			def compute_A(x, u):
				u = u[0]
				A = np.array([[1, self.dt * u_linear[0] * np.cos(x[1] + np.arctan( self.lr * np.tan(u) / self.L ))],\
					[0, 1]])
				return A
			def compute_B(x, u):
				u = u[0]
				c1 = self.lr**2 * np.tan(u)**2 / self.L**2 + 1
				c2 = np.tan(u)**2 + 1
				c3 = np.sin( x[1] + np.arctan( self.lr * np.tan(u) / self.L ) )
				c4 = np.cos( x[1] + np.arctan( self.lr * np.tan(u) / self.L ) )
				B = np.array([[self.dt * self.lr * u_linear[0] * c4 * c2 / (self.L * c1)],\
					[self.dt * u_linear[0] * c2 / (self.L * c1**(1/2)) - self.dt * self.lr**2 * u_linear[0] * np.tan(u)**2 *c2 / (self.L**3 * c1**(3/2))]])
				return B
			self.jfx = lambda x, u: compute_A(x, u)
			self.jfu = lambda x, u: compute_B(x, u)
			self.ode = lambda t, x, u: np.array([[u_linear[0] * np.sin(x[1] + np.arctan(self.lr * np.tan(u[0]) / self.L))], \
				[u_linear[0] * np.tan(u[0]) * np.cos( np.arctan(self.lr * np.tan(u[0]) / self.L) )]]).flatten()


		self.Ad = self.jfx(self.x0, [self.u0])
		self.Bd = self.jfu(self.x0, [self.u0])
		self.Cd = [[1, 0],
			[0, 1]]
		self.Dd = [[0], [0]]
		print(self.Ad)
		print(self.Bd)
		print(self.Dd)

		self.n = len(self.Ad)
		self.m = len(self.Bd[0])
		self.x0 = np.array(self.x0).reshape((self.n, ))
		self.u0 = np.array(self.u0).reshape((self.m, ))
		# self.u_min = np.array( [V_MIN, HEADING_MIN] )
		# self.u_max = np.array( [V_MAX, HEADING_MAX] )
		self.u_min = np.array( u_min )
		self.u_max = np.array( u_max )
		sysd = StateSpace(self.Ad, self.Bd, self.Cd, self.Dd, dt=dt)
		self.Ad = sysd.A
		self.Bd = sysd.B
		self.Cd = sysd.C
		self.Dd = sysd.D
	
