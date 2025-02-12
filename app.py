from flask import Flask, render_template
from flask_socketio import SocketIO
from constants import *
import time
from datetime import datetime, timezone
import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.twobody.propagation import propagate
from system import Satellite
from astropy.time import Time
from dataclasses import dataclass
import sched, time

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
    nb_corps = len(M)
    F = np.zeros((nb_corps * 6, 1))
    for i in range(nb_corps):
        S = np.zeros(3)
        for j in range(nb_corps):
            if j != i:
                S += force(M[i], y[i*3:(i+1)*3, 0], M[j], y[j*3:(j+1)*3, 0])
        F[i*3:(i+1)*3, 0] = y[nb_corps*3 + i*3 : nb_corps*3 + (i+1)*3, 0]
        F[nb_corps*3 + i*3 : nb_corps*3 + (i+1)*3, 0] = S / M[i]
    return F    

def RK4(t0, tf, y, N, M):
    """Runge-Kutta 4th order solver."""
    t = t0
    h = (tf-t0)/N
    while t < tf: 
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

def gauss_jackson(t0, tf, y, N, M):
    """Gauss-Jackson implicit predictor-corrector solver."""
    t = t0
    h = (tf - t0) / N
    # Initial steps using Verlet integration
    y_prev = y - h * f(t, y, M)
    y_curr = y
    y_next = y + h * f(t, y, M)
    
    while t < tf:
        # Predictor step
        y_pred = 2 * y_curr - y_prev + h**2 * f(t, y_curr, M)
        
        # Corrector step
        y_corr = y_curr + 0.5 * h * (f(t, y_curr, M) + f(t + h, y_pred, M))
        
        y_prev = y_curr
        y_curr = y_corr
        t += h
    
    return y_curr

@dataclass
class Simulation:
    r: Satellite

app = Flask(__name__)
socketio = SocketIO(app)
should_shutdown = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('button_create')
def handle_button_clicked(message):
    Y = [-6545, -3490, 2500, -3.457, 6.618, 2.533]

    socketio.emit('800', Y)
    socketio.emit('802', FREQ_SIM)
    socketio.emit('803', FREQ_AFF)

@socketio.on('button_run')
def handle_button_clicked(message):
    Y = np.array([[-6545e3], [-3490e3], [2500e3], [0], [0], [0], [-3.457e3], [6.618e3], [2.533e3], [0], [0], [0]])
    M = np.array([100, 6e24])
    
    start = time.time()
    live_sim = 0
    live_aff = 0
    tic = 1/FREQ_AFF
    end = 20000 #seconds
    step = 1
    send_status = False

    while live_sim <= end:

        t0 = time.time()
        Y = gauss_jackson(0, step, Y, 10, M)
        live_sim += step
        live_aff = time.time() - start
        t1 = time.time()

        if (live_aff <= tic) & (send_status == False):
            socketio.emit('801', [Y[0, 0]/1000, Y[1, 0]/1000, Y[2, 0]/1000])
            socketio.emit('802', min(FREQ_SIM, int(step/(t1-t0))))
            send_status = True

        if live_aff > tic:
            tic += 1/FREQ_AFF
            send_status = False

        if t1-t0 < step/FREQ_SIM:
            time.sleep(step/FREQ_SIM - (t1-t0))

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=1500)
