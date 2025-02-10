from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
from datetime import datetime, timezone

app = Flask(__name__)
socketio = SocketIO(app)
should_shutdown = False

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000)