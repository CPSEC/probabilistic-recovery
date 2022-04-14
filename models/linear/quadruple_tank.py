# Ref: K. H. Johansson, "The quadruple-tank process: a multivariable laboratory process with an adjustable zero," in IEEE Transactions on Control Systems Technology, vol. 8, no. 3, pp. 456-465, May 2000, doi: 10.1109/87.845876.
# Ref: J. Giraldo, S. H. Kafash, J. Ruths and A. A. Cardenas, "DARIA: Designing Actuators to Resist Arbitrary Attacks Against Cyber-Physical Systems," 2020 IEEE European Symposium on Security and Privacy (EuroS&P), 2020, pp. 339-353, doi: 10.1109/EuroSP48549.2020.00029.

import numpy as np 
import math

# parameters
A1 = A3 = 28
A2 = A4 = 32
a1 = a3 = 0.071
a2 = a4 = 0.057
kc = 0.5
g = 981
# with minimum-phase characteristics
T1 = 62
T2 = 90
T3 = 23
T4 = 30
k1 = 3.33
k2 = 3.35
gamma1 = 0.70
gamma2 = 0.60

A = np.array(
    [[-1/T1,    0,       A3/(A1*T3),    0],
     [0,        -1/T2,   0,             A4/(A2*T4)],
     [0,        0,       -1/T3,         0],
     [0,        0,       0,             -1/T4]]
)

B = np.array(
    [[gamma1*k1/A1,     0],
     [0,                gamma2*k2/A2],
     [0,                (1-gamma2)*k2/A3],
     [(1-gamma1)*k1/A4, 0]]
)

C = np.array(
    [[kc, 0,  0, 0],
     [0,  kc, 0, 0]]
)

D =  