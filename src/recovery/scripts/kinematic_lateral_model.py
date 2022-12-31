from math import atan, tan, cos, sin 
import numpy as np

# parameters
delta_r = 0   # steering angle of rear wheel
l_r = 2.852 * 0.5   # distance between mass center and rear wheel
l_f = l_r
v = 5         # velocity
dt = 0.05

class LaneKeeping:
    def ode(self, t, x, u):
        """
        x = [x, y, psi]    # location x, location y, heading angle
        u = [delta_f]      # steering angle of front wheel
        """
        delta_f = u[0]   # steering angle of front wheel
        psi = x[2]
        beta = atan((l_f*tan(delta_r)+l_r*tan(delta_f))/(l_f+l_r))  # slip angle
        dx = v * cos(psi+beta)
        dy = v * sin(psi+beta)
        dpsi = v * cos(beta) * (tan(delta_f)-tan(delta_r)) / (l_f + l_r)
        return np.array([dx, dy, dpsi])

    def lin_dis(self, x, u):
        delta_f = u[0]   # steering angle of front wheel
        psi = x[2]
        Ad=np.array([
            [1, 0, -dt*v*sin(psi + atan((l_f*tan(delta_r) + l_r*tan(delta_f))/(l_f + l_r)))],
            [0, 1,  dt*v*cos(psi + atan((l_f*tan(delta_r) + l_r*tan(delta_f))/(l_f + l_r)))],
            [0, 0,                                                                        1]])
        Bd=np.array([
            [                                                                                                                       -dt*l_r*v*(tan(delta_f)**2 + 1)*sin(psi + atan((l_f*tan(delta_r) + l_r*tan(delta_f))/(l_f + l_r)))/((1 + (l_f*tan(delta_r) + l_r*tan(delta_f))**2/(l_f + l_r)**2)*(l_f + l_r))],
            [                                                                                                                        dt*l_r*v*(tan(delta_f)**2 + 1)*cos(psi + atan((l_f*tan(delta_r) + l_r*tan(delta_f))/(l_f + l_r)))/((1 + (l_f*tan(delta_r) + l_r*tan(delta_f))**2/(l_f + l_r)**2)*(l_f + l_r))],
            [dt*(-l_r*v*(l_f*tan(delta_r) + l_r*tan(delta_f))*(tan(delta_f) - tan(delta_r))*(tan(delta_f)**2 + 1)/((1 + (l_f*tan(delta_r) + l_r*tan(delta_f))**2/(l_f + l_r)**2)**(3/2)*(l_f + l_r)**3) + v*(tan(delta_f)**2 + 1)/(sqrt(1 + (l_f*tan(delta_r) + l_r*tan(delta_f))**2/(l_f + l_r)**2)*(l_f + l_r)))]])
        cc=(x + self.ode(0, x, u)*dt) - Ad@x - Bd@u
        return Ad, Bd, cc

    
    

