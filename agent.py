from dataclasses import dataclass
import numpy as np
from constants import *

@dataclass
class Motion:
    """
    motion = [[x0, y0, z0, vx0, vy0, vz0],
              [x1, y1, z1, vx1, vy1, vz1],
                        ...
              [xn, yn, zn, vxn, vyn, vzn]]

    """

    motion: np.ndarray
    
@dataclass
class Object:
    id: int
    motion = [[EARTH_RADIUS+400,EARTH_RADIUS+400,EARTH_RADIUS+400,0,0,0]]

    def __post_init__(self):
        return