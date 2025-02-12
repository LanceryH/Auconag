from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
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

app = Flask(__name__)
socketio = SocketIO(app)
should_shutdown = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('button_create')
def handle_button_clicked(message):
    r = [-6045, -3490, 2500] * u.km
    v = [-3.457, 6.618, 2.533] * u.km / u.s
    global initial
    initial = Orbit.from_vectors(Earth, r, v)

    socketio.emit('800', initial.r.to_value(u.km).tolist())

@socketio.on('button_run')
def handle_button_clicked(message):
    final = initial.propagate(30 * u.min)

    socketio.emit('801', final.r.to_value(u.km).tolist())

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=1500)










"""
tart = time.time()
epoch = Time("2024-01-01 00:00:00", scale="utc")
a = u.Quantity(7000, u.km)  # Semi-major axis
ecc = u.Quantity(0.01)  # Eccentricity
inc = u.Quantity(45, u.deg)  # Inclination
raan = u.Quantity(80, u.deg)  # Right ascension of the ascending node
argp = u.Quantity(30, u.deg)  # Argument of periapsis
nu = u.Quantity(0, u.deg)  # True anomaly

orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

global sat
sat = Satellite(orbit)
sat_p = sat.orbit.r

print("Satellite creation took:", time.time() - start, "seconds")

sat_pos = sat_p.to_value(u.km).tolist()

print("Satellite creation took:", time.time() - start, "seconds")

socketio.emit('800', sat.orbit.r.to_value(u.km).tolist())
    """