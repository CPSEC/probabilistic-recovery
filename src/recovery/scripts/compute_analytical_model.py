import sympy as sym
from sympy import atan, tan, sin, cos
import numpy as np
from copy import deepcopy

# ----------  modify -----------------------------
x, y, psi, l_f, l_r, delta_f, delta_r, v = sym.symbols('x y psi l_f l_r delta_f delta_r v')
beta = atan((l_f*tan(delta_r)+l_r*tan(-delta_f))/(l_f+l_r))  # slip angle
dx = v * cos(psi+beta)
dy = v * sin(psi+beta)
dpsi = v * cos(beta) * (tan(-delta_f)-tan(delta_r)) / (l_f + l_r)

vars = [x, y, psi]
us = [delta_f]
f = [dx, dy, dpsi]
# ----------------------------------------------------

dt = sym.symbols('dt')
J = sym.zeros(len(f), len(vars))
for i, fi in enumerate(f):
    for j, s in enumerate(vars):
        J[i, j] = sym.diff(fi, s)
Ac = deepcopy(J)
Ad = sym.eye(len(vars)) + dt * Ac
print(f'{Ad=}')

J = sym.zeros(len(f), len(us))
for i, fi in enumerate(f):
    for j, s in enumerate(us):
        J[i, j] = sym.diff(fi, s)
Bc = deepcopy(J)
Bd = dt * Bc
print(f'{Bd=}')




