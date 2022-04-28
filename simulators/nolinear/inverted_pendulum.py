# Ref: Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control (Session 8.7, Page 300)

import numpy as np
from scipy.linalg import solve_continuous_are, inv
from utils import Simulator

# parameters
m = 1     # mass of rob
M = 5     # mass of cart
L = 2     # length of rob
g = -10
d = 1     # dumping (friction)
b = 1     # pendulum up (b=1)

def inverted_pendulum(t, x, u, params={}):
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m*L*L*(M+m*(1-Cx*Cx))

    dx = np.zeros((4,))
    dx[0] = x[1]
    dx[1] = (1/D)*(-m*m*L*L*g*Cx*Sx + m*L*L*(m*L*x[3]*x[3]*Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*x[3]*x[3]*Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx


# control parameters
A = np.array([[0, 1, 0, 0],
              [0, -d/M, b*m*g/M, 0],
              [0, 0, 0, 1],
              [0, -b*d/(M*L), -b*(m+M)*g/(M*L), 0]])
B = np.array([[0], [1/M], [0], [b*1/(M*L)]])

R = np.array([[0.0001]])
Q = np.eye(4)

P = np.matrix(solve_continuous_are(A, B, Q, R))
K = np.matrix(inv(R) * (B.T * P))


class InvertedPendulum(Simulator):
    """
    States: (4,)
        x[0]: location of cart
        x[1]: dx[0]
        x[2]: pendulum angle  (down:0, up:pi)
        x[3]: dx[1]
    Control Input: (1,)
        u[0]: force on the cart
    Output: (4,)
        State Feedback
    Controller: LQR
    """
    def __init__(self):
        pass





