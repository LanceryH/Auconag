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
    
    epoch = Time.now()
    a = 7000 * u.km  # Semi-major axis
    ecc = 0.01 * u.one  # Eccentricity
    inc = 45 * u.deg  # Inclination
    raan = 80 * u.deg  # Right ascension of the ascending node
    argp = 30 * u.deg  # Argument of periapsis
    nu = 0 * u.deg  # True anomaly

    orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

    global sat
    sat = Satellite(orbit)

    socketio.emit('800', sat.orbit.r.to_value(u.km).tolist())

@socketio.on('button_run')
def handle_button_clicked(message):

    new_orbit = sat.propagate(6)

    socketio.emit('801', new_orbit.r.to_value(u.km).tolist())

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=1500)

