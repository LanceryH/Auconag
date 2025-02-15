from dataclasses import dataclass
import time
from constants import *
from agent_class import *

@dataclass
class Simulation:
    agent: Agent
    start = time.time()
    live_sim = 0
    live_aff = 0
    tic = 1/FREQ_AFF
    step = 1
    send_status = False
    freq_sim_max = 0

    def update(self):
        self.live_aff = time.time() - self.start
        self.live_sim += self.step
        self.freq_sim_max = self.agent.train_me()
        if self.live_aff > self.tic:
                self.tic += 1/FREQ_AFF
        if self.freq_sim_max < self.step/FREQ_SIM:
                time.sleep(self.step/FREQ_SIM - self.freq_sim_max)

    @property
    def is_ready_to_send(self):
        return (self.live_aff <= self.tic) & (not self.send_status)