import numpy as np
from classes.dynamic_class import *
from constants import *

def force(m1, X1, m2, X2):
    """Compute the gravitational force between two masses."""
    G = 6.674e-11
    d = np.linalg.norm(X1 - X2)
    if d == 0:
        return np.zeros(3)  # Avoid division by zero
    f = - (G * m1 * m2 / d**3) * (X1 - X2)
    return f

def f(t, y, M):
    """Computes the derivatives for the system of equations."""
    N = len(M)
    F = np.zeros_like(y)
    for i in range(N):
        S = np.zeros(3)
        for j in range(N):
            if j != i:
                S += force(M[i], y[i][:3], M[j], y[j][:3])
        F[i][:3] = y[i][3:6]
        F[i][3:6] = S / M[i] + y[i][6:9]
    return F    

def RK4(y, N, M):
    """Runge-Kutta 4th order solver."""
    t = 0
    h = SIM_STEP/N
    while t < SIM_STEP: 
        k1 = h * f(t, y, M)
        k2 = h * f(t + h/2, y + k1/2, M)
        k3 = h * f(t + h/2, y + k2/2, M)
        k4 = h * f(t + h, y + k3, M)
        y = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h    
    return y

def verlet(t0, tf, y, N, M):
    """Verlet integration solver."""
    t = t0
    h = (tf - t0) / N
    y_prev = y - h * f(t, y, M)  # Initial step using Euler method
    while t < tf:
        y_next = 2 * y - y_prev + h**2 * f(t, y, M)
        y_prev = y
        y = y_next
        t += h
    return y

def gauss_jackson(y, M):
    """Gauss-Jackson implicit predictor-corrector solver."""
    y = np.array(y)
    t = 0
    h = SIM_STEP / SIM_N
    # Initial steps using Verlet integration
    y_prev = y - h * f(t, y, M)
    y_curr = y
    y_next = y + h * f(t, y, M)
    
    while t < SIM_STEP:
        y_pred = 2 * y_curr - y_prev + h**2 * f(t, y_curr, M)
        y_corr = y_curr + 0.5 * h * (f(t, y_curr, M) + f(t + h, y_pred, M))
        y_prev = y_curr
        y_curr = y_corr
        t += h
    
    return Dynamic(pos=y_curr[0][:3], vel=y_curr[0][3:6], acc=y_curr[0][6:9], mass=M[0]), Dynamic(pos=y_curr[1][:3], vel=y_curr[1][3:6], acc=y_curr[1][6:9], mass=M[1])