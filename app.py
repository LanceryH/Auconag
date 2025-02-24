from flask import Flask, render_template
from flask_socketio import SocketIO
from constants import *
from classes.dynamic_class import *
from system import *
from classes.agent_class import *
import numpy as np
import time
import numpy as np
from classes.simulation_class import *
from classes.body_class import *
import multiprocessing as mp

app = Flask(__name__)
socketio = SocketIO(app)
should_shutdown = False
simulation = Simulation()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('button_create')
def handle_button_clicked(message):
    simulation.add_earth()
    simulation.add_moon()
    socketio.emit('700', simulation.state)
    return

@socketio.on('button_run')
def handle_button_clicked(message):
    simulation.add_agents(NB_AGENTS)
    socketio.emit('700', simulation.state)

    simulation.initiate()
    while simulation.is_active:
        simulation.update()

        if simulation.is_ready_to_send:
            socketio.emit('700', simulation.state)
            socketio.emit('800', simulation.infos)

@socketio.on('button_test')
def handle_button_clicked(message):
    return

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=1500)
