from simulators.nonlinear.continuous_stirred_tank_reactor import cstr
from scipy.optimize import fsolve
import numpy as np

x_ss = np.array([0.98189, 300.00013])
u_ss = np.array([274.57786])
ode = cstr
ode_with_fixed_x = lambda u: ode(0, x_ss, u)
print(f'{ode_with_fixed_x(u_ss)=}')
u_ss = fsolve(ode_with_fixed_x, u_ss)
print(f'{u_ss=}')