from dataclasses import dataclass

class Simulation:
    def __init__(self, bodies=None, physics=None):
        self.bodies = bodies if bodies else []
        self.physics = physics  # Optional Physics object

    def __repr__(self):
        return f"Simulation(bodies={self.bodies}, physics={self.physics})"

    def __add__(self, other):
        if isinstance(other, Body):  
            return Simulation(self.bodies + [other], self.physics)  # Add body, keep physics
        elif isinstance(other, Physics):  
            return Simulation(self.bodies, other)  # Set physics
        return NotImplemented

@dataclass
class Body:
    x: float
    y: float
    z: float
    m: float

    def __add__(self, other):
        if isinstance(other, Body):  
            return Simulation([self, other])  # Start a new Simulation
        elif isinstance(other, Simulation):  
            return Simulation([self] + other.bodies, other.physics)  # Add to existing Simulation
        return NotImplemented

@dataclass
class Physics:
    G: float
    live: bool

# Example Usage
body1 = Body(1, 2, 3, 4)
body2 = Body(5, 6, 7, 8)
body3 = Body(9, 10, 11, 12)

# Physics
physics = Physics(6.67430e-11, False)

simulation = body1 + body2 + body3 + physics

print(simulation)  
physics.live = False
print(simulation)  

# Output: Simulation(bodies=[Body(x=1, y=2, z=3, m=4), Body(x=5, y=6, z=7, m=8), Body(x=9, y=10, z=11, m=12)], physics=Physics(G=6.6743e-11, live=True))
