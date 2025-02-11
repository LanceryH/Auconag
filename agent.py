from dataclasses import dataclass
import numpy as np

@dataclass
class System:
    """
    motion = [[x0, y0, z0, vx0, vy0, vz0],
              [x1, y1, z1, vx1, vy1, vz1],
                        ...
              [xn, yn, zn, vxn, vyn, vzn]]

    """

    motion: np.ndarray

    @property
    def pos_x(self):
        return self.motion[0,:]
    @property
    def pos_y(self):
        return self.motion[1,:]
    @property
    def pos_z(self):
        return self.motion[2,:]
    @property
    def vel_x(self):
        return self.motion[3,:]
    @property
    def vel_y(self):
        return self.motion[4,:]
    @property
    def vel_z(self):
        return self.motion[5,:]
    
@dataclass
class Object:
    position = (0, 0)