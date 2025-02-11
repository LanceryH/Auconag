from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
from datetime import datetime, timezone
from agent import Object

app = Flask(__name__)
socketio = SocketIO(app)
should_shutdown = False

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('button_clicked')
def handle_button_clicked(message):
    rocket = Object(0)
    print(f'Received message: {message} \n rocket is {rocket.motion}')


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.112', port=2000)