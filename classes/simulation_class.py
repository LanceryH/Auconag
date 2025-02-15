from dataclasses import dataclass
from classes.agent_class import *
from constants import *
from classes.body_class import *
import time

def freq_ctrl(func):
    def wrapper(arg: 'Simulation'):
        t0 = time.time()
        func(arg)
        t1 = time.time()
        arg.freq_sim_max = t1 - t0
    return wrapper

@dataclass
class Simulation:
    agent: Agent
    body: Body
    start = time.time()
    live_sim = 0
    live_aff = 0
    tic = 1/FREQ_AFF
    step = 1
    send_status = False
    freq_sim_max = 0

    @freq_ctrl
    def update(self):
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