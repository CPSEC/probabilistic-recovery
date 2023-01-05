
# Import numpy and scipy
from numpy import pi
import numpy as np
from scipy.signal import StateSpace
# Import Rtss
from recovery.utils.formal.gaussian_distribution import GaussianDistribution
from recovery.utils.formal.reachability import ReachableSet
from recovery.utils.formal.zonotope import Zonotope
from recovery.utils.observers.kalman_filter import KalmanFilter
from recovery.utils.formal.strip import Strip
# Import system model
class RTSS:
    
    def __init__(self, Ad, Bd, Cd, Dd, W, u_min, u_max, k_reconstruction, k_max, l, a, b):
        '''
        Inputs
        A, B, C, D: system matrices
        W: Noise matrix
        u_min, u_max: minimum and maximum control
        k_reconstruction: maximum states that can be reconstructed
        k_max: maximum number of steps to compute the reconstruction k_reconstruction >= k_max
        '''
        assert k_reconstruction >= k_max
        self.Ad = Ad
        self.Bd = Bd
        self.Cd = Cd
        self.Dd = Dd
        # Create zonotope
        self.U = Zonotope.from_box(u_min, u_max)
        # Create reachable set
        self.k_max = k_max
        self.reach = ReachableSet(Ad, Bd, self.U, W, max_step=k_reconstruction)
        # Create strip
        self.s = Strip(l, a=a, b=b)
        # Create kalman filter
        self.kf = None
        self.x_cur_update = None
    
    def set_kalman_filter(self, C, Q, R):
        self.kf = KalmanFilter(self.Ad, self.Bd, C, self.Dd, Q, R)
    
    
    def recovery_isolation_fs(self, us_checkpoint, ys_checkpoint, x_checkpoint):
        '''
        Assumes set_kalman_filter has been called before
        States have also been stored
        '''
        us_checkpoint = np.array(us_checkpoint)
        ys_checkpoint = np.array(ys_checkpoint)
        x_res, P_res = self.kf.multi_steps(x_checkpoint, np.zeros_like(self.Ad), us_checkpoint, ys_checkpoint)
        x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
        self.x_cur_update = x_cur_update
        self.reach.init(x_cur_update, self.s)
        k, X_k, D_k, z_star, alpha, P, arrive = self.reach.given_k(max_k=self.k_max)
        rec_u_temp = self.U.alpha_to_control(alpha)
        return rec_u_temp, k
    
    def recovery_isolation_ns(self, cur_x, cur_u):
        '''
        States have also been stored
        '''
        cur_u = np.array(cur_u).reshape( (len(self.Bd[0]), ) )
        cur_x = np.array(cur_x).reshape( (len(self.Ad[0]), ) )
        x_cur_predict = GaussianDistribution(*self.kf.predict(self.x_cur_update.miu, self.x_cur_update.sigma, cur_u))
        y = self.kf.C @ cur_x
        # print("current state:", cur_x)
        # print("current meas:", y)
        # print("update: ", self.x_cur_update)
        # print("predict: ", x_cur_predict)
        x_cur_update = GaussianDistribution(*self.kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
        self.x_cur_update = x_cur_update
        self.reach.init(x_cur_update, self.s)
        # print("Attack detected, state:", self.x_cur_update.miu)
        k, X_k, D_k, z_star, alpha, P, arrive = self.reach.given_k(max_k=self.k_max)
        rec_u_temp = self.U.alpha_to_control(alpha)
        # print("Probability:", P)
        return rec_u_temp[0], k
        
    
    def recovery_no_isolation(self, us, x_0):
        us = np.array(us)
        n = len(self.Ad)
        x_0 = GaussianDistribution( x_0, np.zeros((n, n)) )
        self.reach.init( x_0, self.s )
        self.x_res_point = self.reach.state_reconstruction(us)
        # print("estimated state:  ", self.x_res_point.miu + np.array([0,  0, -1,  0,  0,  0,   1,   0,   0,   0,   1,   0,   0,   0,   1,  0,  0,  0] ))
        self.reach.init( self.x_res_point, self.s )
        # Run one-step rtss
        k, X_k, D_k, z_star, alpha, P, arrive = self.reach.given_k(max_k=self.k_max)
        # Compute u
        rec_u = self.U.alpha_to_control(alpha)
        # print("Probability:", P)
        return rec_u, k

