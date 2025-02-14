from dataclasses import dataclass, field

@dataclass
class Dynamic:
    pos: list
    vel: list
    acc: list
    mass: float

    @property
    def state(self):
        return [*self.pos, *self.vel, *self.acc]

    def update_to(self, new_dynamic: 'Dynamic'):
        self.pos = new_dynamic.pos
        self.vel = new_dynamic.vel
        self.acc = new_dynamic.acc
        self.mass = new_dynamic.mass
