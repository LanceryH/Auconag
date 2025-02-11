from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate
import numpy as np
import matplotlib.pyplot as plt

class Satellite:
    def __init__(self, orbit: Orbit):
        """Initialize the satellite with an orbit."""
        self.orbit = orbit

    def propagate(self, time_span_hours):
        """Propagates the satellite's orbit for a given number of hours."""
        time_span = time_span_hours * u.hour
        future_orbit = self.orbit.propagate(time_span)
        return future_orbit