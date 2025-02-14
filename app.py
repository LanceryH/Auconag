from flask import Flask, render_template
from flask_socketio import SocketIO
from dynamic import *
from constants import *
from system import *
from agent import *
import numpy as np
import time
import numpy as np
    
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
    Y = np.array([[-6545e3, -3490e3, 2500e3, -3.457e3, 6.618e3, 2.533e3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    M = np.array([100, 6e24])
    
    start = time.time()
    live_sim = 0
    live_aff = 0
    tic = 1/FREQ_AFF
    end = 2000 #seconds
    step = 1
    send_status = False

    while live_sim <= end:

        t0 = time.time()
        Y = gauss_jackson(0, step, Y, 10, M)
        live_sim += step
        live_aff = time.time() - start
        t1 = time.time()

        if (live_aff <= tic) & (send_status == False):
            socketio.emit('801', (Y[0][:3]).tolist())
            socketio.emit('802', min(FREQ_SIM, int(step/(t1-t0))))
            send_status = True

        if live_aff > tic:
            tic += 1/FREQ_AFF
            send_status = False

        if t1-t0 < step/FREQ_SIM:
            time.sleep(step/FREQ_SIM - (t1-t0))

@socketio.on('button_test')
def handle_button_clicked(message):
    agent_dyn = Dynamic(pos=[-6545e3, -3490e3, 2500e3],
                        vel=[-3.457e3, 6.618e3, 2.533e3],
                        acc=[0, 0, 0],
                        mass=100)

    agent = DQNAgent(agent_dyn, state_dim=9, action_dim=3)

    for episode in range(EPISODES):
        socketio.emit('803', FREQ_AFF)

        start = time.time()
        live_sim = 0
        live_aff = 0
        tic = 1/FREQ_AFF
        step = 1
        send_status = False
        
        while agent.active:

            t0 = time.time()
            live_sim += step
            live_aff = time.time() - start
            agent.train_me()
            t1 = time.time()

            if (live_aff <= tic) & (send_status == False):
                socketio.emit('801', agent.dynamic.state[:3])
                socketio.emit('802', min(FREQ_SIM, int(step/(t1-t0))))
    
                send_status = True

            if live_aff > tic:
                tic += 1/FREQ_AFF
                send_status = False

            if t1-t0 < step/FREQ_SIM:
                time.sleep(step/FREQ_SIM - (t1-t0))
                
        print(f"Episode {episode+1}: Total Reward: {agent.total_reward}")

        agent_dyn = Dynamic(pos=[-6545e3, -3490e3, 2500e3],
                            vel=[-3.457e3, 6.618e3, 2.533e3],
                            acc=[0, 0, 0],
                            mass=100)
                            
        agent.update_target_network(agent_dyn)

    return agent
    
    
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=1500)
