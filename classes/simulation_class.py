from dataclasses import dataclass
from classes.agent_class import *
from constants import *
from classes.body_class import *
import time
from random import randint

def freq_ctrl(func):
    def wrapper(arg: 'Simulation'):
        t0 = time.time()
        func(arg)
        t1 = time.time()
        arg.freq_sim_max = t1 - t0
    return wrapper

def random_3d_position(min_norm=6371):
    position = np.random.normal(size=3)  # Generate a random vector from a normal distribution
    position /= np.linalg.norm(position)  # Normalize it to unit length
    position *= np.random.uniform(min_norm, min_norm + 10000)  # Scale it to be above min_norm
    return position*1e3

def random_3d_velocity(min_speed=7.3, max_speed=8.2):
    velocity = np.random.normal(size=3)  # Generate a random 3D direction
    velocity /= np.linalg.norm(velocity)  # Normalize to unit length
    velocity *= np.random.uniform(min_speed, max_speed)  # Scale to a realistic orbital speed
    return velocity*1e3

@dataclass
class Simulation:
    live_sim = 0
    live_aff = 0
    tic = 1/FREQ_AFF
    step = 1
    send_status = False
    freq_sim_max = 0

    def __post_init__(self):
         self.agents: list[Agent] = []

    def initiate(self):
         self.start = time.time()

    def add_agents(self, n):
        for _ in range(n):
            pos = random_3d_position()
            vel = random_3d_velocity()
            self.agents.append(Agent(Dynamic(pos, vel, acc=[0, 0, 0], mass=100)))

    def add_earth(self):
         self.earth = Body(Dynamic(pos=[0, 0, 0], vel=[0, 0, 0], acc=[0, 0, 0], mass=6e24))
    
    def add_moon(self):
        self.moon = Body(Dynamic(pos=[384.4e6, 0, 0], vel=[0, 1.022e3, 0], acc=[0, 0, 0], mass=7.35e22))


    @property
    def infos(self):
        return {"live_sim": self.live_sim, 
                "live_aff": self.live_aff, 
                "freq_sim_max": int(1/self.freq_sim_max),
                "nb_agents": len(self.agents)}

    @property
    def state(self):
        return {"agents": [agent.dynamic.state for agent in self.agents], 
                "earth": self.earth.dynamic.state, 
                "moon": self.moon.dynamic.state}

    
    @property
    def is_active(self):
        return self.live_sim <= SIM_TIME_MAX
    
    @freq_ctrl
    def update(self):
        self.live_sim += self.step
        self.live_aff = time.time() - self.start


        earth_new_dyn, moon_new_dyn = gauss_jackson([self.earth.dynamic.state, self.moon.dynamic.state],
                                                    [self.earth.dynamic.mass, self.moon.dynamic.mass])
        
        self.earth.dynamic.update_to(earth_new_dyn)
        self.moon.dynamic.update_to(moon_new_dyn)

        for agent in self.agents:
            Y = [agent.dynamic.state, self.earth.dynamic.state, self.moon.dynamic.state]
            M = [agent.dynamic.mass, self.earth.dynamic.mass, self.moon.dynamic.mass]

            agent_new_dyn, _, _ = gauss_jackson(Y, M)
            agent.dynamic.update_to(agent_new_dyn)

        if self.live_aff > self.tic:
                self.tic += 1/FREQ_AFF
                
        if self.freq_sim_max < self.step/FREQ_SIM:
                time.sleep(self.step/FREQ_SIM - self.freq_sim_max)

            
    @freq_ctrl
    def updatesss(self):
        self.live_aff = time.time() - self.start
        self.live_sim += self.step
        self.agent.train()

        thrust_x = 0
        thrust_y = 0
        thrust_z = 0

        if self.agent.action == 0:
            thrust_x = 150

        elif self.agent.action == 1:
            thrust_y = 150
            
        elif self.agent.action == 2:
            thrust_z = 150

        self.agent.dynamic.acc = [thrust_x, thrust_y, thrust_z]

        agen_new_dyn, body_new_dyn = gauss_jackson([self.agent.dynamic.state, self.body.dynamic.state],
                                                   [self.agent.dynamic.mass, self.body.dynamic.mass])

        self.agent.dynamic.update_to(agen_new_dyn)
        self.body.dynamic.update_to(body_new_dyn)

        if self.live_aff > self.tic:
                self.tic += 1/FREQ_AFF
                
        if self.freq_sim_max < self.step/FREQ_SIM:
                time.sleep(self.step/FREQ_SIM - self.freq_sim_max)

    @property
    def is_ready_to_send(self):
        return (self.live_aff <= self.tic) & (not self.send_status)

    @property
    def freq_minimal(self):
        return min(FREQ_SIM, int(self.step/self.freq_sim_max))