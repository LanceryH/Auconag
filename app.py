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
    # Define the agent initial dynamic
    simulation = Simulation()
    simulation.add_agent()
    simulation.add_earth()
    simulation.add_moon()

    agent1 = Agent(Dynamic(pos=[-6545e3, -3490e3, 2500e3],
                           vel=[-3.457e3, 6.618e3, 2.533e3],
                           acc=[0, 0, 0],
                           mass=100))
    
    agent2 = Agent(Dynamic(pos=[-6745e3, -3490e3, 2500e3],
                           vel=[-3.457e3, 6.618e3, 2.533e3],
                           acc=[0, 0, 0],
                           mass=100))
    
    # Define the earth initial dynamic
    earth = Body(Dynamic(pos=[0, 0, 0],
                         vel=[0, 0, 0],
                         acc=[0, 0, 0],
                         mass=6e24))
    
    # Define the moon initial dynamic
    moon = Body(Dynamic(pos=[384.4e6, 0, 0], 
                        vel=[0, 1.022e3, 0],    
                        acc=[0, 0, 0],
                        mass=7.35e22)) 
    
    agents = [agent1, agent2]

    start = time.time()
    live_sim = 0
    live_aff = 0
    tic = 1/FREQ_AFF
    end = 2000 #seconds
    step = 1
    send_status = False

    while live_sim <= end:

        t0 = time.time()

        for agent in agents:
            agent_new_dyn, earth_new_dyn, moon_new_dyn = gauss_jackson([agent.dynamic.state, earth.dynamic.state, moon.dynamic.state],
                                                                       [agent.dynamic.mass, earth.dynamic.mass, moon.dynamic.mass])


            agent.dynamic.update_to(agent_new_dyn)
        earth.dynamic.update_to(earth_new_dyn)
        moon.dynamic.update_to(moon_new_dyn)

        live_sim += step
        live_aff = time.time() - start
        t1 = time.time()

        if (live_aff <= tic) & (send_status == False):
            socketio.emit('700', agent1.dynamic.state[:3])
            socketio.emit('701', earth.dynamic.state[:3])
            socketio.emit('702', moon.dynamic.state[:3])
            socketio.emit('800', int(1/(t1-t0)))

            send_status = True

        if live_aff > tic:
            tic += 1/FREQ_AFF
            send_status = False

        if t1-t0 < step/FREQ_SIM:
            time.sleep(step/FREQ_SIM - (t1-t0))

@socketio.on('button_test')
def handle_button_clicked(message):
    # Define the agent initial dynamic
    agent = Agent(Dynamic(pos=[-6545e3, -3490e3, 2500e3],
                          vel=[-3.457e3, 6.618e3, 2.533e3],
                          acc=[0, 0, 0],
                          mass=100))
    
    # Define the earth initial dynamic
    earth = Body(Dynamic(pos=[0, 0, 0],
                         vel=[0, 0, 0],
                         acc=[0, 0, 0],
                         mass=6e24))
    
    # Define the simulation
    simul = Simulation(agent, earth)

    for episode in range(EPISODES):
        while agent.active:
            simul.update()

            if simul.is_ready_to_send:
                socketio.emit('801', agent.dynamic.state)
                socketio.emit('802', simul.freq_minimal)
                
        print(f"Episode {episode+1}: Total Reward: {agent.total_reward}")

        # Restart the agent dynamic
        dynamic = Dynamic(pos=[-6545e3, -3490e3, 2500e3],
                          vel=[-3.457e3, 6.618e3, 2.533e3],
                          acc=[0, 0, 0],
                          mass=100)
                            
        agent.restart_me(dynamic)
    
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=1500)
    socketio.emit('803', FREQ_AFF)