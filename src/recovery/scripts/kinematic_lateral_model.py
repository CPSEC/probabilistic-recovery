from math import atan, tan, cos, sin, sqrt
import numpy as np
from state_record import StateRecord

# parameters
delta_r = 0   # steering angle of rear wheel
l_r = 0.4 #2.852 * 0.5   # distance between mass center and rear wheel
l_f = 3.1
v = 5         # velocity
dt = 0.05

class Kinematic:
    def __init__(self) -> None:
        self.n = 3    # number of states
        self.m = 1    # number of control input
        self.data = StateRecord()

    def ode(self, t, x, u):
        """
        x = [x, y, psi]    # location x, location y, heading angle
        u = [delta_f]      # steering angle of front wheel
        """
        delta_f = -u[0]   # steering angle of front wheel
        psi = x[2]
        beta = atan((l_f*tan(delta_r)+l_r*tan(delta_f))/(l_f+l_r))  # slip angle
        dx = v * cos(psi+beta)
        dy = v * sin(psi+beta)
        dpsi = v * cos(beta) * (tan(delta_f)-tan(delta_r)) / (l_f + l_r)
        return np.array([dx, dy, dpsi])

    def f(self, x, u):
        dx = self.ode(None, x, u)
        return x+dt * dx

    def jfx(self, x, u):
        """
        got from src/recovery/scripts/compute_analytical_model.py
        """
        delta_f = u[0]   # steering angle of front wheel
        psi = x[2]
        Ad=np.array([
            [1, 0, -dt*v*sin(psi + atan((l_f*tan(delta_r) - l_r*tan(delta_f))/(l_f + l_r)))],
            [0, 1,  dt*v*cos(psi + atan((l_f*tan(delta_r) - l_r*tan(delta_f))/(l_f + l_r)))],
            [0, 0,                                                                        1]])
        return Ad

    def jfu(self, x, u):
        delta_f = u[0]   # steering angle of front wheel
        psi = x[2]       
        Bd=np.array([
            [                                                                                                                         dt*l_r*v*(tan(delta_f)**2 + 1)*sin(psi + atan((l_f*tan(delta_r) - l_r*tan(delta_f))/(l_f + l_r)))/((1 + (l_f*tan(delta_r) - l_r*tan(delta_f))**2/(l_f + l_r)**2)*(l_f + l_r))],
            [                                                                                                                        -dt*l_r*v*(tan(delta_f)**2 + 1)*cos(psi + atan((l_f*tan(delta_r) - l_r*tan(delta_f))/(l_f + l_r)))/((1 + (l_f*tan(delta_r) - l_r*tan(delta_f))**2/(l_f + l_r)**2)*(l_f + l_r))],
            [dt*(l_r*v*(l_f*tan(delta_r) - l_r*tan(delta_f))*(-tan(delta_f) - tan(delta_r))*(tan(delta_f)**2 + 1)/((1 + (l_f*tan(delta_r) - l_r*tan(delta_f))**2/(l_f + l_r)**2)**(3/2)*(l_f + l_r)**3) + v*(-tan(delta_f)**2 - 1)/(sqrt(1 + (l_f*tan(delta_r) - l_r*tan(delta_f))**2/(l_f + l_r)**2)*(l_f + l_r)))]])
        return Bd



