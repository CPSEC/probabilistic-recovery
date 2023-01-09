import sympy as sym
from sympy import atan, tan, sin, cos
import numpy as np
import math
from copy import deepcopy

# ----------  modify -----------------------------
q, V, rho, Cp, mdelH, EoverR, k0, UA, Tc, Ca, T, Caf, Tf  = sym.symbols('q V rho Cp mdelH EoverR k0 UA Tc Ca T Caf Tf')

rA = k0 * sym.exp(-EoverR / T) * Ca
# Calculate concentration derivative
dCadt = q / V * (Caf - Ca) - rA
# Calculate temperature derivative
dTdt = q / V * (Tf - T) \
       + mdelH / (rho * Cp) * rA \
       + UA / V / rho / Cp * (Tc - T)

vars = [Ca, T]
us = [Tc]
f = [dCadt, dTdt]
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




