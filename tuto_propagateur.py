# coding=utf-8
import time

start = time.time()
import copy
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from astropy.time import Time
from astropy import units as u
import simulator_constants
from poliastro.bodies import Earth, Moon
from poliastro.constants import H0_earth

from poliastro.core.elements import rv2coe
from poliastro.core.perturbations import (
    atmospheric_drag,
    third_body,
    J2_perturbation,
    J3_perturbation,
)
from poliastro.earth.atmosphere import COESA76
from poliastro.core.propagation import func_twobody
from poliastro.ephem import build_ephem_interpolant
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator


import utils


class Satellite:
    def __init__(
        self,
        name="Satellite",
        mass=850,
        area=15,
        cd=2.2, # coef de trainée 2.2 valeur moyenne
        a=6728.137,
        ecc=0,
        inc=0,
        raan=0,
        argp=0,
        ta=0,
        epoch=Time(2451545.0, format="jd", scale="tdb"), # epoch par défaut à J2000
    ):

        self.coesa = COESA76()  # Modèle de densité atmosphérique COESA 76
        self.H0 = H0_earth.to(u.km).value

        # Définition des paramètres par défault du satellite
        self.name = name
        self.mass = mass  # Masse [kg] (défault = 850 kg)
        self.area = area  # Surface [m**2] (défaut = 15 m**2)
        self.cd = cd  # Coefficient de traînée [sans unité] (défault = 2.2)

        # Initialisation de l'orbite avec POLIASTRO
        self.sat = Orbit.from_classical(
            attractor=Earth,
            a=self.add_unit(a, u.km),
            ecc=self.add_unit(ecc, u.one),
            inc=self.add_unit(inc, u.deg),
            raan=self.add_unit(raan, u.deg),
            argp=self.add_unit(argp, u.deg),
            nu=self.add_unit(ta, u.deg),
            epoch=epoch,
        )
        
        self.start_epoch = self.epoch
        self.moon = self.initialize_moon()
        # Initialisation des vecteurs de données
        # vecteur de taille (N,6) contenant le vecteur d'état dans le repère ECI [X,Y,Z,VX,VY,VZ] (ECI = on s'en fout de la rotation de la Terre)
        self.x_eci = np.concatenate((self.r.value, self.v.value)).reshape((6, 1))

        # vecteur de taille (N,7) contenant les éléments képlériens [a,h,ecc,inc,raan,argp,ta]
        self.kep = np.array(
            [
                self.a.value,
                self.h.value,
                self.ecc.value,
                self.inc.value,
                self.raan.value,
                self.argp.value,
                self.ta.value,
            ]
        ).reshape((7, 1))

        self.times = [0]

    def add_unit(self, var, unit):
        """This function makes sure a variable (var) has a unit (unit)

        Args:
            var (float or int): any variable
            unit (astropy.unit): unit form astropy librairy

        Returns:
            flaot or int with unit: return the varialble with the unit
        """
        if not isinstance(var, u.Quantity):
            return var * unit
        else:
            return var

    @property
    def r(self):
        """Vecteur position ECI [km]"""
        return self.sat.r

    @property
    def v(self):
        """Vecteur vitesse ECI [km/s]"""
        return self.sat.v

    @property
    def a(self):
        """Demi-grand axe [km]"""
        return self.sat.a

    @property
    def h(self):
        """Altitude [km]"""
        return np.linalg.norm(self.r) - simulator_constants.EARTH_RADIUS * u.km

    @property
    def ecc(self):
        """Excentricité [degré]"""
        return self.sat.ecc

    @property
    def inc(self):
        """Inclinaison [degré]"""
        return self.sat.inc.to(u.deg)

    @property
    def raan(self):
        """Longitude du noeud ascendant [degré]"""
        return self.sat.raan.to(u.deg)

    @property
    def argp(self):
        """Argument du périgée [degré]"""
        return self.sat.argp.to(u.deg)

    @property
    def ta(self):
        """Anomalie vraie [degré]"""
        return self.sat.nu.to(u.deg)

    @property
    def epoch(self):
        """Epoch [julian date]"""
        return self.sat.epoch

    @property
    def n(self):
        """Mouvement moyen [rad/s]"""
        return self.sat.n

    @property
    def period(self):
        """Période orbitale [s]"""
        return self.sat.period

    @property
    def r_a(self):
        """Apoapsis théorique [s]"""
        return self.sat.r_a

    @property
    def r_p(self):
        """Périapsis théorique [s]"""
        return self.sat.r_p

    def initialize_moon(self):
        """This functions initializes the object Moon from Poliastro. It goes from the start epoch to 60 days later.

        Returns:
            Poliastro bodies : Moon (start_epoch → start_epoch + 60 days)
        """
        moon = build_ephem_interpolant(
            Moon,
            28 * u.day,
            (
                self.start_epoch.value * u.day,
                self.start_epoch.value * u.day + 60 * u.day,
            ),
            rtol=1e-2,
        )
        return moon

    def rv2coe(self):
        """Convertit le vecteur d'état ECI en éléments orbitaux 
        """
        return rv2coe(simulator_constants.MU_EARTH, self.r, self.v)

    def rv(self):
        """Integration of the rv function from Poliastro into the class
            
        Returns:
            array with quantities :[[x,y,z] << (u.km), [vx,vy,vz] << (u.km/u.s)]
        """
        return self.sat.rv()
    
    def print_orbital_elements(self):
        """Print the orbital elements : a, ecc, inc, raan, argp, ta
        """
        print(f"a = {np.round(self.a, 3)} km, ecc = {np.round(self.ecc, 6)}, inc = {np.round(self.inc, 3)} °, raan = {np.round(self.raan, 3)} °, argp = {np.round(self.argp, 3)} °, ta = {np.round(self.ta, 3)} °")

    def copy(self, copy_data=True):
        """This function does a copy of the satellite.\n
        It is also possible to copy or not all the data (position ECI and times)
        
        Utile pour faire des tests sur un satellite 
        On réalise les tests sur la copie du satellite sans changer l'original
        Args:
            copy_data (bool, optional): if True all the datat are copied in the satellite copy. Defaults to True.

        Returns:
            Satellite object: Satellite copy
        """
        new_sat = Satellite(
            name=copy.deepcopy(self.name),
            mass=copy.deepcopy(self.mass),
            area=copy.deepcopy(self.area),
            cd=copy.deepcopy(self.cd),
            a=copy.deepcopy(self.a.to(u.km)),
            ecc=copy.deepcopy(self.ecc.to(u.one)),
            inc=copy.deepcopy(self.inc.to(u.deg)),
            raan=copy.deepcopy(self.raan.to(u.deg)),
            argp=copy.deepcopy(self.argp.to(u.deg)),
            ta=copy.deepcopy(self.ta.to(u.deg)),
            epoch=copy.deepcopy(self.epoch),
        )
        # If copy_data is True, we copy all the data in the satellite copy (x_eci, kep and times)
        # The start epoch of the copy is the satellite's start_epoch
        # The moon is initialized from start_epoch to 60 days later
        if copy_data:
            new_sat.start_epoch = copy.deepcopy(self.start_epoch)
            new_sat.x_eci = copy.deepcopy(self.x_eci)
            new_sat.kep = copy.deepcopy(self.kep)
            new_sat.times = copy.deepcopy(self.times)
            new_sat.moon = new_sat.initialize_moon()

        # If copy_data is False, the copy's start_epoch is the current satellite's epoch
        # The Moon is initialized from epoch to 60 days later
        return new_sat

    def from_vectors(self, r, v, epoch=None):
        """This function can modify the state vector of the satellite by defining a new position and/or velocity\n
        In the class, it is used to modify the velocity of the satellite by adding a delta V

        Args:
            r (array (3,1) or list): Position vector
            v (array (3,1) or list): Velocity vector
            epoch (astropy Time epoch): epoch of the satellite, By default satellite.epoch
        """
        if epoch is None:
            epoch = self.epoch

        self.sat = Orbit.from_vectors(Earth, r, v, epoch=epoch)

        # Updating the satellite's keplerian elements
        self.kep[:, -1] = np.array(
            [
                self.a.value,
                self.h.value,
                self.ecc.value,
                self.inc.value,
                self.raan.value,
                self.argp.value,
                self.ta.value,
            ]
        ).reshape((7,))

    def model_coesa_atmospheric_density(self, rv_satellite):
        """This function use the atmospheric model COESA76 from Poliastro \n
        to model the density of the atmosphere under 1000 km of altitude\n
        Above 1000 km, the air density is computed by using a simplified model from Poliastro.perturbation\n

        Args:
            rv_satellite (array (6,1) or list): state vector of the satellite [x,y,z,vx,vy,vz]

        Returns:
            float: rho, the air density at the altitude's satellite
        """

        altitude = (
            np.linalg.norm(np.array(rv_satellite[:3]))
            - simulator_constants.EARTH_RADIUS
        )
        if altitude < 1000:
            rho = self.coesa.density(alt=(altitude) * u.km).value
        else:
            rho = simulator_constants.RHO0 * np.exp(-(altitude) / self.H0)
        return rho * 1e9

    def propagator(self, t0, state, k):
        """This function compute the acceleration of the satellite at time : satellite's epoch + t0 (seconds)
        It takes into account atmospheric drag, J2 harmonic, J3 harmonic and the Moon
        It is used in the propagate Poliastro function with the CowellPropagator

        Si tu veux modifier le propagateur, tu as juste à enlever à garder les accélérations de ton choix 
        Args:
            t0 (float or int): time in second from the satellite's epoch
            state (array (6,1) or list): state vector of the satellite [x,y,z,vx,vy,vz]
            k (float): standard gravitation parameter (Earth : 398600.44180000003)

        Returns:
            array (6,1): accelerationa array [0,0,0,ax,ay,az]
        """

        du_kep = func_twobody(t0, state, k)# accélération induite par la Terre
        ax, ay, az = atmospheric_drag(
            t0,
            state,
            k,
            C_D=self.cd,
            A_over_m=((self.area) * (u.m**2) / (self.mass * u.kg)).to_value(
                u.km**2 / u.kg
            ),
            rho=self.model_coesa_atmospheric_density(state), 
        )
        du_drag = np.array([0, 0, 0, ax, ay, az])# acc induite par l'atmosphere

        ax, ay, az = J2_perturbation(
            t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
        )
        du_j2 = np.array([0, 0, 0, ax, ay, az]) # acc induite par le J2 (applatissement des poles)

        ax, ay, az = J3_perturbation(
            t0, state, k, J3=Earth.J3.value, R=Earth.R.to(u.km).value
        )
        du_j3 = np.array([0, 0, 0, ax, ay, az]) # acc induite par le J3

        ax, ay, az = third_body(
            (self.epoch + t0 * u.s - self.start_epoch).to(u.s).value,
            state,
            k,
            k_third=Moon.k.to(u.km**3 / u.s**2).value,
            perturbation_body=self.moon,
        )
        du_moon = np.array([0, 0, 0, ax, ay, az]) # acc induite par la Lune

        return du_kep + du_drag + du_j2 + du_j3 + du_moon

    def propagate(self, t, dt=None, n=None):
        """This function propagate the satellite from is current epoch to t (seconds)
        If dt and n are None, it propagates to t without computing intermediate values
        Else, it propagates to t with a step time of dt or a numer of step times n

        Args:
            t (float or int): propagation time in seconds (without astropy unit)\n
            dt (float or int , optional): value of the step time in second (without astropy unit). Defaults to None.\n
            n (int, optional): number of step time. Defaults to None.
        """
        if dt is None and n is None:
            # Propagation to t
            self.sat = self.sat.propagate(
                t * u.s, method=CowellPropagator(f=self.propagator)
            )
            # Add the time
            self.times.append((self.epoch - self.start_epoch).to(u.s).value)
            # Add the new position ECI
            self.x_eci = np.hstack(
                (
                    self.x_eci,
                    np.concatenate((self.r.value, self.v.value)).reshape((6, 1)),
                )
            )
            # Add the new position keplerian elements
            self.kep = np.hstack(
                (
                    self.kep,
                    np.array(
                        [
                            self.a.value,
                            self.h.value,
                            self.ecc.value,
                            self.inc.value,
                            self.raan.value,
                            self.argp.value,
                            self.ta.value,
                        ]
                    ).reshape((7, 1)),
                )
            )

        # If dt or n are specifies we compute the list of step time withe the utils function
        else:
            t0 = self.times[-1]
            t_tot = t0 + t
            for step_time_value in utils.get_list_of_time_step(0, t, dt=dt, n=n):
                # Propagation of step_time_value
                self.sat = self.sat.propagate(
                    step_time_value * u.s, method=CowellPropagator(f=self.propagator)
                )
                # Adding time value to time vector
                self.times.append((self.epoch - self.start_epoch).to(u.s).value)
                print(f"\r{self.times[-1]}/{t_tot}", end="")

                # Add the new position ECI
                self.x_eci = np.hstack(
                    (
                        self.x_eci,
                        np.concatenate((self.r.value, self.v.value)).reshape((6, 1)),
                    )
                )
                # Add the new position keplerian elements
                self.kep = np.hstack(
                    (
                        self.kep,
                        np.array(
                            [
                                self.a.value,
                                self.h.value,
                                self.ecc.value,
                                self.inc.value,
                                self.raan.value,
                                self.argp.value,
                                self.ta.value,
                            ]
                        ).reshape((7, 1)),
                    )
                )

    def just_propagate(self, t):
        """This function can propagate the satellite to time t without saving the data\n
          and modify the state of the satellite.
          It is used in several functions to do some tests on the satellite
            
        Args:
            t (float or int): time of propagation in second (whitout astropy unit)

        Returns:
            Poliastro orbit object: satellite propagated to time t
        """
        return self.sat.propagate(t * u.s, method=CowellPropagator(f=self.propagator))

    def get_real_orbital_period(self, precision=1e-3):
        """This function finds the real orbital period.
        It uses the function minimize_scalar from scipy that is initialized with the theoretical period value

        Args:
            precision (float, optional): Orbital period value precision. Defaults to 1e-3.

        Raises:
            ValueError: Raise if the function does not converge

        Returns:
            float: Real orbital period value
        """
        initial_position = self.r.value

        def position_difference_at_time(time_of_propagation):
            satellite_position = self.just_propagate(time_of_propagation).r.value
            return np.linalg.norm(satellite_position - initial_position)

        # Function initialisation: result between (0.95*T, 1.05*T) with T the theoretical period value
        result = minimize_scalar(
            position_difference_at_time,
            bounds=(self.period.to(u.s).value * 0.95, self.period.to(u.s).value * 1.05),
            method="bounded",
            options={"xatol": precision, "maxiter": 1000},
        )
        if result.success:
            return result.x
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def propagation_time_to_altitude(self, target_altitude, precision=1e-5):
        """This function finds the moment where the satellite's altitude is equal to the target_altitude.\n
        It uses the function minimize_scalar from scipy to finds the value by minimization

        Args:
            target_altitude (float or int): The target altitude (without astropy unit)
            precision (_type_, optional): _description_. Defaults to 1e-5.

        Raises:
            ValueError: Raise an error if the function does not converge

        Returns:
            float or int : time of propagation where the satellite is at the target altitude
            float : real altitude achieved by the satellite
        """
        target_radius = target_altitude + simulator_constants.EARTH_RADIUS

        def altitude_difference_at_time(time_of_propagation):
            satellite_position = self.just_propagate(time_of_propagation).r.value
            current_radius = np.linalg.norm(satellite_position)
            return abs(current_radius - target_radius)

        result = minimize_scalar(
            altitude_difference_at_time,
            bounds=(0, self.period.value),
            method="bounded",
            options={"xatol": precision},
        )
        if result.success:
            time_of_propagation = result.x
            return (
                time_of_propagation,
                np.linalg.norm(self.just_propagate(time_of_propagation).r.value)
                - simulator_constants.EARTH_RADIUS,
            )
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def get_periapsis(self, precision=1e-6):
        """This function finds the real periapsis of the satellite. \n
        It uses the function minimize_scalar from scipy.

        Args:
            precision (float, optional): Precision of the time of propagation to reach the periapsis. Defaults to 1e-6.

        Returns:
            float or int : time of propagation where the satellite is at his periapsis
            float : The real periapsis (distance norm from the center of Earth)
        """

        def altitude(time_of_propagation):
            return np.linalg.norm(self.just_propagate(time_of_propagation).r.value)

        result = minimize_scalar(
            altitude,
            bounds=(0, self.period.value),
            method="bounded",
            options={"xatol": precision},
        )
        if result.success:
            return result.x, np.linalg.norm(self.just_propagate(result.x).r.value)
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def get_apoapsis(self):
        """This function finds the real apoapsis of the satellite. \n
        It uses the function propagation_time_to_altitude, with the target_altitude: satellite.r_a
        This is the theoretical value of the apoapsis and du to the perturbation, the real one is lower\n
        We choose this way, because it is faster than the method used in the function get_periapsis \n
        applied to the apoapsis

        Returns:
            float or int : time of propagation where the satellite is at his apoapsis
            float : The real apoapsis (distance norm from the center of Earth)
        """
        time_of_propagation, real_r_a = self.propagation_time_to_altitude(
            self.r_a.value - simulator_constants.EARTH_RADIUS
        )
        return time_of_propagation, real_r_a + simulator_constants.EARTH_RADIUS

    def get_tlong(self):
        """This function compute the true longitude:
        TLONG = (RAAN + ARGP + TA) % 360

        This is used in the phasing maneuver.
        Returns:
            float: True Longitude in degrees (without astropy unit)
        """
        return (
            self.raan.to(u.deg).value
            + self.argp.to(u.deg).value
            + self.ta.to(u.deg).value
        ) % 360

    def propagation_time_to_angle(self, target_angle, precision=1e-4):
        """This function finds the time of propagation where the True longitude of the satellite\n
          will be the same as the target_angle.
        It uses the function minimize_scalar from scipy and it is initialized using theoretical value.
        Initialisation :
        Time_initialisation = (|target_angle-current_angle| % 360) / satellite_angular_velocity

        Args:
            target_angle (float or int): The target angle
            precision (float, optional): Precision of the time of propagation. Defaults to 1e-4.

        Raises:
            ValueError: Raise an error if the function does not converge

        Returns:
            float: time of propagation where the satellite's is equal to the target_angle
        """
        # Initilisation computed in radian
        target_angle = np.deg2rad(target_angle)  # rad
        w_sat = np.sqrt(simulator_constants.MU_EARTH / self.a.value**3)  # rad/s
        current_angle = np.deg2rad(self.get_tlong())
        tof_approx = ((target_angle - current_angle) % (2 * np.pi)) / w_sat

        def angle_difference_at_time(time_of_propagation):
            # Copy the satellite to propagate whitout modify the state
            new_sat = self.copy()
            new_sat.propagate(time_of_propagation)
            current_angle = np.deg2rad(new_sat.get_tlong())
            return abs(current_angle - target_angle)

        result = minimize_scalar(
            angle_difference_at_time,
            bounds=(tof_approx * 0.9, tof_approx * 1.1),
            method="bounded",
            options={"xatol": precision},
        )
        if result.success:
            return result.x
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def impulsive_burn_to_period(self, target_period, precision=1e-5):
        """This function calculates the delta V required for the orbital period \n
        to be equal to the target period.
        It uses the function minimize_scalar from scipy and it is initialized with the theory

        Initialisation:
        new_sma = (MU_EARTH * (target_period/(2*pi))**2)**(1/3)
        new_velocity = (2 * ((MU_EARTH / |r|) - (MU_EARTH / (2*new_sma))))**(1/2)
        |delta_V_init| = new_velocity - current_velocity

        This is used especially in the phasing maneuver.

        Args:
            target_period (float or int): The target orbital period
            precision (float, optional): Precision of the delta V value (magnitude). Defaults to 1e-5.

        Raises:
            ValueError: Raise an error if the function does not converge

        Returns:
            list : list of the delta V vector ([dVx, dVy, dVz] << (u.km/u.s))
        """
        r, v = self.rv()

        # Compute the angle between each axes (ECI frame) and the velocity vector
        # The delta V will have the same angles to not modify too much the orbit
        norm_vec = np.linalg.norm(v.value)
        angle_x = np.arccos(v.value[0] / norm_vec)
        angle_y = np.arccos(v.value[1] / norm_vec)
        angle_z = np.arccos(v.value[2] / norm_vec)

        # Initialisation using the theory
        new_sma = np.cbrt(
            simulator_constants.MU_EARTH * (target_period / (2 * np.pi)) ** 2
        )
        new_v = np.sqrt(
            2
            * (
                (simulator_constants.MU_EARTH / np.linalg.norm(r.value))
                - (simulator_constants.MU_EARTH / (2 * new_sma))
            )
        )
        current_v = np.linalg.norm(v.value)
        delta_v_init = new_v - current_v
        if delta_v_init < 0:
            bounds = (1.1 * delta_v_init, 0.9 * delta_v_init)
        else:
            bounds = (0.9 * delta_v_init, 1.1 * delta_v_init)

        def period_difference(dv):
            deltav = [
                dv * np.cos(angle_x),
                dv * np.cos(angle_y),
                dv * np.cos(angle_z),
            ] << (u.km / u.s)
            new_sat = self.copy()
            new_sat.from_vectors(r, v + deltav, epoch=new_sat.epoch)

            # After adding the delta V, if the difference between the theoretical value of the orbital period
            # and the target_period is lower than 1500, we compute the real value with the function get_real_orbital_period
            # This step saves a few seconds of execution time.
            if abs(new_sat.period.value - target_period) > 1500:
                new_period = new_sat.period.value
            else:
                new_period = new_sat.get_real_orbital_period()
            return abs(new_period - target_period)

        result = minimize_scalar(
            period_difference,
            bounds=bounds,
            method="bounded",
            options={"xatol": precision},
        )
        if result.success:
            dv = result.x
            return [dv * np.cos(angle_x), dv * np.cos(angle_y), dv * np.cos(angle_z)]
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def impulse_to_altitude(self, target_altitude, target_time_to_altitude):
        """This function computes the delta V required to reach a target altitude (target_altitude)\n
        in a given time (target_time_to_altitude).
        It uses the function minimize from scipy and it is initialized with the theory
        The initialisation, is used to know if the satellite need to decelerate (delta V < 0) \n
        or accelerate (delta V > 0)

        Initialisation:
        new_sma = (|current_r|+|target_altitude + EARTH_RADIUS|) / 2
        new_velocity = (2 * ((MU_EARTH / |current_r|) - (MU_EARTH / (2*new_sma))))**(1/2)
        |delta_V_init| = new_velocity - current_velocity

        Args:
            target_altitude (flaot or int): The target altitude
            target_time_to_altitude (float or int): Propagation time after which the satellite\n
            reaches the target altitude

        Raises:
            ValueError: Raise an error if the function does not converge

        Returns:
            list : list of the delta V vector ([dVx, dVy, dVz] << (u.km/u.s))
        """
        r, v = self.rv()
        norm_vec = np.linalg.norm(v.value)
        angle_x = np.arccos(v.value[0] / norm_vec)
        angle_y = np.arccos(v.value[1] / norm_vec)
        angle_z = np.arccos(v.value[2] / norm_vec)

        current_r = np.linalg.norm(r.value)
        target_r = target_altitude + simulator_constants.EARTH_RADIUS
        new_sma = (current_r + target_r) / 2
        current_v = np.linalg.norm(v.value)
        new_v = np.sqrt(simulator_constants.MU_EARTH * ((2 / current_r) - 1 / new_sma))
        delta_v_init = new_v - current_v

        print("Initialisation : |delta V init| = ", delta_v_init)
        if delta_v_init < 0:
            bounds = (-3, 0)
        elif delta_v_init > 0:
            bounds = (0, 3)
        else:
            bounds = (-3, 3)

        def altitude_difference(x):
            dv = x[0]
            deltav = [
                dv * np.cos(angle_x),
                dv * np.cos(angle_y),
                dv * np.cos(angle_z),
            ] << (u.km / u.s)
            new_sat = self.copy()
            new_sat.from_vectors(r, v + deltav)
            pos = new_sat.just_propagate(target_time_to_altitude).r.value
            return abs(
                np.linalg.norm(pos) - simulator_constants.EARTH_RADIUS - target_altitude
            )

        result = minimize(
            altitude_difference,
            x0=[delta_v_init],
            method="Nelder-Mead",
            bounds=[bounds],
            options={"xatol": 1e-6, "disp": 3},
        )
        if result.success:
            dv = result.x[0]
            print(f"Result : Delta V = {dv} ({np.linalg.norm(dv)} km/s)")
            return [
                dv * np.cos(angle_x),
                dv * np.cos(angle_y),
                dv * np.cos(angle_z),
            ] << (u.km / u.s)
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def impulsive_burn_to_target_apogee(self, target_apoapsis, precision=1e-5):
        """This function calculates the impulsive delta V required to modify\n
          the orbital apoapsis to the target apoapsis.
        It uses the function minimize_scalar from scipy and it is initialized with the theory

        Initialisation:
        new_sma = (|current_r|+|target_altitude + EARTH_RADIUS|) / 2
        new_velocity = (2 * ((MU_EARTH / |current_r|) - (MU_EARTH / (2*new_sma))))**(1/2)
        |delta_V_init| = new_velocity - current_velocity

        Args:
            target_altitude (flaot or int): The target altitude
            target_time_to_altitude (float or int): Propagation time after which the satellite\n
            reaches the target altitude

        Raises:
            ValueError: Raise an error if the function does not converge

        Returns:
            list : list of the delta V vector ([dVx, dVy, dVz] << (u.km/u.s))
        """
        r, v = self.rv()
        norm_vec = np.linalg.norm(v.value)
        angle_x = np.arccos(v.value[0] / norm_vec)
        angle_y = np.arccos(v.value[1] / norm_vec)
        angle_z = np.arccos(v.value[2] / norm_vec)

        current_r = np.linalg.norm(r.value)
        target_r = target_apoapsis + simulator_constants.EARTH_RADIUS
        new_sma = (current_r + target_r) / 2
        current_v = np.linalg.norm(v.value)
        new_v = np.sqrt(simulator_constants.MU_EARTH * ((2 / current_r) - 1 / new_sma))
        delta_v_init = new_v - current_v

        print("Initialisation : |delta V init| = ", delta_v_init)
        if delta_v_init < 0:
            bounds = (2 * delta_v_init, 0.3 * delta_v_init)
        else:
            bounds = (0.3 * delta_v_init, 2 * delta_v_init)

        new_sat = self.copy()

        def apoapsis_difference(dv):
            deltav = [
                dv * np.cos(angle_x),
                dv * np.cos(angle_y),
                dv * np.cos(angle_z),
            ] << (u.km / u.s)
            new_sat.from_vectors(r, v + deltav)

            # To save time, we compare the theoretical apoapsis value which is slightly different from the real value.
            # If the difference between the theoretical value and the target value is lower than 200 km
            # We compute the real apoapsis value
            apogee_achieved = new_sat.r_a.value - simulator_constants.EARTH_RADIUS
            if abs(apogee_achieved - target_apoapsis) < 200:
                _, apogee_achieved = new_sat.propagation_time_to_altitude(
                    new_sat.r_a.value - simulator_constants.EARTH_RADIUS
                )
            return abs(apogee_achieved - target_apoapsis)

        result = minimize_scalar(
            apoapsis_difference,
            bounds=bounds,
            method="bounded",
            options={"xatol": precision},
        )
        if result.success:
            dv = result.x
            print(f"Result : Delta V = {dv} ({np.linalg.norm(dv)} km/s)")
            return [dv * np.cos(angle_x), dv * np.cos(angle_y), dv * np.cos(angle_z)]
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def impulsive_burn_to_ecc(self, target_ecc, precision=1e-6):
        """This function calculates the delta V required to modify the orbital eccentricity value.\n
        It is used to circularize the orbit.
        It uses the function minimize_scalar from scipy.

        Args:
            target_ecc (float): The target eccentricity
            precision (float, optional): The precision of the delta V magnitude. Defaults to 1e-6.

        Raises:
            ValueError: Raise an error if the function does not converge

        Returns:
            list : list of the delta V vector ([dVx, dVy, dVz] << (u.km/u.s))
        """

        r, v = self.rv()
        norm_vec = np.linalg.norm(v.value)
        angle_x = np.arccos(v.value[0] / norm_vec)
        angle_y = np.arccos(v.value[1] / norm_vec)
        angle_z = np.arccos(v.value[2] / norm_vec)

        def ecc_to_target(dv):
            deltav = [
                dv * np.cos(angle_x),
                dv * np.cos(angle_y),
                dv * np.cos(angle_z),
            ] << u.km / u.s
            new_sat = Orbit.from_vectors(Earth, r, v + deltav, epoch=self.epoch)
            return abs(target_ecc - new_sat.ecc.value)

        result = minimize_scalar(
            ecc_to_target,
            bounds=(-3, 3),
            method="bounded",
            options={"xatol": precision},
        )
        if result.success:
            dv = result.x
            print(f"Result : Delta V = {dv} ({np.linalg.norm(dv)} km/s)")
            return [dv * np.cos(angle_x), dv * np.cos(angle_y), dv * np.cos(angle_z)]
        else:
            raise ValueError("La méthode de minimisation n'a pas convergé.")

    def hohmann_transfer(self, target_apoapsis, target_ecc):
        """This function computes the time and the magnitude of the two maneuvers of the Hohman transfer

        The first maneuvers needs to be realised at the periapsis,\n
        so we get the time of propagation to the periapsis with the function get_periapsis.
        The first maneuver, delta_v1, modify the apoapsis of the orbit to be equal at the target_apoapsis,\n
        computed with the function impulsive_burn_to_target_apogee.
        The second maneuvers is realised when the satellite reachs the apoapsis,\n
        and it circuralises the orbit with the function impulsive_burn_to_ecc

        Args:
            target_altitude (float or int )
            target_ecc (float )

        Returns:
            list: list containing the times of both maneuvers [time_delta_v1, time_delta_v2]
            list: list containing the delta V of both maneuvers [delta_v1, delta_v2]

        """
        print(f"Hohmann transfer from {self.h} km to {target_apoapsis} km")
        # propagation time to periapsis
        time_delta_v1, _ = self.get_periapsis()

        new_sat = self.copy()
        new_sat.sat = new_sat.just_propagate(time_delta_v1)
        r_sat, v_sat = new_sat.rv()
        print(
            f"Periapsis reached at {np.linalg.norm(r_sat.value) - 6378.137} after t={time_delta_v1} s"
        )

        # Compute the delta V1 to modify the apoapsis
        delta_v1 = new_sat.impulsive_burn_to_target_apogee(target_apoapsis) << (
            u.km / u.s
        )
        new_sat.from_vectors(r_sat, v_sat + delta_v1, epoch=new_sat.epoch)
        time_delta_v2, _ = new_sat.propagation_time_to_altitude(
            new_sat.r_a.value - simulator_constants.EARTH_RADIUS
        )

        new_sat.sat = new_sat.just_propagate(time_delta_v2)

        # Compute de delta V2 to circularise the orbit
        delta_v2 = new_sat.impulsive_burn_to_ecc(target_ecc=target_ecc) << (u.km / u.s)

        return [time_delta_v1, time_delta_v2], [delta_v1, delta_v2]


if __name__ == "__main__":

    start = time.time()
    test = Satellite(
        name="Satellite",
        a=7178,
        inc=0,
        ecc=0.000004,
        raan=145,
        argp=54,
        ta=60,
    )
    
    # Affichage des données 
    test.print_orbital_elements()
    print(test.rv())

    # test de proagation pdt 1 jour avec un pas de temps de 60 secondes
    test.propagate(t=86400, dt=600)

    # Calcul de la période réelle 
    # print(test.get_real_orbital_period())

    # Transfert de Hohmann jusqu'en GEO altitude de 36000 km avec une orbite finale circulaire ecc=0
    times_h, dv_h = test.hohmann_transfer(target_apoapsis=36000, target_ecc=0)
    print(times_h, dv_h)
    
    # Propagation du transfert d'Hohmann
    test.propagate(times_h[0]) # Propagation jusqu'au pérgigée
    test.from_vectors(test.r, test.v + dv_h[0]) # Manoeuvre 1 : orbite de tansfert
    test.propagate(times_h[1]) # Propagation du temps de vol 
    test.from_vectors(test.r, test.v + dv_h[1]) # Maneouvre 2 : circularisation de l'orbite
    test.propagate(test.period.value / 2, n=100) # Facultatif : propagation d'une demie période avec 100 points

    # Manoeuvre pour se placer à une altitude de 36200 km en 6000s
    dv_altitude = test.impulse_to_altitude(36200, 6000) # Calcule du delta V
    test.from_vectors(test.r, test.v + dv_altitude) # Application de la manoeuvre
    test.propagate(6000, n=100) # Propagation de 6000s
    print(test.h)

