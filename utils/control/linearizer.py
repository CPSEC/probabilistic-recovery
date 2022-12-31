from scipy.optimize import approx_fprime
import numpy as np
from functools import partial
from scipy.signal import StateSpace

class Linearizer:
    def __init__(self, ode, nx, nu, dt, jfx=None, jfu=None):
        self.ode = ode
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.jfx = jfx
        self.jfu = jfu

    def at(self, x_0: np.ndarray, u_0: np.ndarray):
        # ode(t, x, u)
        if self.jfx == None or self.jfu == None:
            A = approx_fprime(x_0, lambda x: self.ode(0, x, u_0))
            B = approx_fprime(u_0, lambda u: self.ode(0, x_0, u))
            C = np.diag([1]*len(A)); D = np.zeros(B.shape) # not important or useful!
            sysc = StateSpace(A, B, C, D)
            self.sysd = sysc.to_discrete(self.dt)
            Ad = self.sysd.A
            Bd = self.sysd.B
        else:
            Ad = self.jfx(x_0, u_0)
            Bd = self.jfu(x_0, u_0)
            Cd = np.diag([1]*len(Ad)); Dd = np.zeros(Bd.shape) # not important or useful!
            self.sysd = StateSpace(Ad, Bd, Cd, Dd)
        assert Ad.shape == (self.nx, self.nx)
        assert Bd.shape == (self.nx, self.nu)
        
        self.sysd.c = (x_0 + self.ode(0, x_0, u_0)*self.dt) - Ad@x_0 - Bd@u_0
        return self.sysd

def analytical_linearize_cstr(x, u, Ts):
    # Ad= Matrix([[Ts*(-1.0 - 72000000000.0*exp(-8750/x1)) + 1, -630000000000000.0*Ts*x0*exp(-8750/x1)/x1**2], [15062761506276.2*Ts*exp(-8750/x1), Ts*(1.31799163179916e+17*x0*exp(-8750/x1)/x1**2 - 3.09205020920502) + 1]])
    Ad = np.empty((2, 2))
    Ad[0, 0] = Ts * (-1.0 - 72000000000.0 * np.exp(-8750 / x[1])) + 1
    Ad[0, 1] = -630000000000000.0 * Ts * x[0] * np.exp(-8750 / x[1]) / x[1] ** 2
    Ad[1, 0] = 15062761506276.2 * Ts * np.exp(-8750 / x[1])
    Ad[1, 1] = Ts * (1.31799163179916e+17 * x[0] * np.exp(-8750 / x[1]) / x[1] ** 2 - 3.09205020920502) + 1

    # Bd= Matrix([[0], [2.09205020920502*Ts]])
    Bd = np.empty((2, 1))
    Bd[0, 0] = 0
    Bd[1, 0] = 2.09205020920502 * Ts

    # Cd= Matrix([[Ts*(x0*(1.0 + 72000000000.0*exp(-8750/x1)) - 1.0*x0 - 72000000000.0*x0*exp(-8750/x1) + 630000000000000.0*x0*exp(-8750/x1)/x1 + 1.0)], [Ts*(x1*(-1.31799163179916e+17*x0*exp(-8750/x1)/x1**2 + 3.09205020920502) - 3.09205020920502*x1 + 350.0)]])
    Cd = np.empty((2,))
    Cd[0] = Ts * (x[0] * (1.0 + 72000000000.0 * np.exp(-8750 / x[1])) - 1.0 * x[0] - 72000000000.0 * x[0] * np.exp(
        -8750 / x[1]) + 630000000000000.0 * x[0] * np.exp(-8750 / x[1]) / x[1] + 1.0)
    Cd[1] = Ts * (x[1] * (
                -1.31799163179916e+17 * x[0] * np.exp(-8750 / x[1]) / x[1] ** 2 + 3.09205020920502) - 3.09205020920502 *
                     x[1] + 350.0)
    return Ad, Bd, Cd

if __name__ == '__main__':
    from simulators.nonlinear.continuous_stirred_tank_reactor import cstr
    linearize = Linearizer(cstr, nx=2, nu=1)
    x_0 = np.array([1, 300])
    u_0 = np.array([280])
    Ad, Bd, cd = linearize.at(x_0, u_0)
    print(f'{Ad=}, \n{Bd=}, \n{cd=}')

