import numpy as np
from scipy.interpolate import interp1d


def get_list_of_time_step(t_start, t_end, dt=None, n=None):
    """This function generates the list of delta times between t_start and t_end using a specified step time (dt)\n
      or a number of step (n)
    Example : get_list_of_time_step(0, 1000, dt=200, n=None) returns [200,200,200,200,200]
    get_list_of_time_step(0, 1000, dt=None, n=5) returns [200,200,200,200,200]

    Args:
        t_start (float): begining time
        t_end (float): ending time
        dt (float or int, optional): step time value. Defaults to None.
        n (int, optional): number of steps. Defaults to None.

    Raises:
        ValueError: Raise if dt and n are both values
        ValueError: Raise if dt and n are both None

    Returns:
        list: list of time step
    """
    if dt is not None and n is not None:
        raise ValueError("Veuillez spécifier soit dt soitn, mais pas les deux.")

    if n is not None:
        dt = (t_end - t_start) / n
        t_vector = [dt] * n
    elif dt is not None:
        n = int((t_end - t_start) // dt)  # Calcul de nsi dt est fourni
        t_vector = [dt] * n
        dtf = (t_end - t_start) - sum(t_vector)  # Calcul du reste à ajouter
        if dtf > 0:
            t_vector.append(dtf)
    else:
        raise ValueError("Veuillez spécifier soit dt soitn.")

    # Correction finale pour garantir que la somme est exacte
    total_time = sum(t_vector)
    print(total_time)
    if total_time > (t_end - t_start):  # Seulement corriger si nécessaire
        correction = (t_end - t_start) - total_time
        t_vector[-1] += correction  # Ajuste le dernier élément

    return t_vector


def interpolate_satellite_position(satellite, t_start, t_end, n):
    """This function interpolates the satellite position by generating n positions between t_start and t_end

    Args:
        satellite (Poliastro Orbit object): a Poliastro satellite
        t_start (float or int): begining time
        t_end (float or int): ending time
        n (int): number of points

    Returns:
        list : list of positions's times
        array (3,n) : array containing the interpolated positions
    """
    interp_funcs = {
        "x": interp1d(satellite.times, satellite.x_eci[0, :], kind="cubic"),
        "y": interp1d(satellite.times, satellite.x_eci[1, :], kind="cubic"),
        "z": interp1d(satellite.times, satellite.x_eci[2, :], kind="cubic"),
    }

    def get_interpolated_position(t):
        return np.array(
            [interp_funcs["x"](t), interp_funcs["y"](t), interp_funcs["z"](t)]
        ).reshape((3, 1))

    pos_interpolated = np.array(satellite.x_eci[:3, 0]).reshape((3, 1))
    t = t_start
    times_vector = [t_start]
    for dt in get_list_of_time_step(t_start, t_end, n=n):
        t += dt
        times_vector.append(t)
        pos_interpolated = np.hstack((pos_interpolated, get_interpolated_position(t)))
    return times_vector, pos_interpolated


def interpolate_satellite_state(satellite, t_start, t_end, n):
    """This function interpolates the state vector (position and velocity) of a satellite by generating n positions \n
    between t_start and t_end

    Args:
        satellite (Poliastro Orbit object): a Poliastro satellite
        t_start (float or int): begining time
        t_end (float or int): ending time
        n (int): number of points

    Returns:
        list : list of positions's times
        array (6,n) : array containing the interpolated positions
    """
    interp_funcs = {
        "x": interp1d(satellite.times, satellite.x_eci[0, :], kind="cubic"),
        "y": interp1d(satellite.times, satellite.x_eci[1, :], kind="cubic"),
        "z": interp1d(satellite.times, satellite.x_eci[2, :], kind="cubic"),
        "vx": interp1d(satellite.times, satellite.x_eci[3, :], kind="cubic"),
        "vy": interp1d(satellite.times, satellite.x_eci[4, :], kind="cubic"),
        "vz": interp1d(satellite.times, satellite.x_eci[5, :], kind="cubic"),
    }

    def get_interpolated_state(t):
        return np.array(
            [
                interp_funcs["x"](t),
                interp_funcs["y"](t),
                interp_funcs["z"](t),
                interp_funcs["vx"](t),
                interp_funcs["vy"](t),
                interp_funcs["vz"](t),
            ]
        ).reshape((6, 1))

    x_interpolated = np.array(satellite.x_eci[:, 0]).reshape((6, 1))
    t = t_start
    times_vector = [t_start]
    for dt in get_list_of_time_step(t_start, t_end, n=n):
        t += dt
        times_vector.append(t)
        x_interpolated = np.hstack((x_interpolated, get_interpolated_state(t)))
    return times_vector, x_interpolated


def compute_matrix_eci2rtn(rv_target):
    """This function computes the matrixes c and c_dot for the conversion from ECI to RTN

    Args:
        rv_target (array (6,1) or list): state vector ECI of the target satellite [x,y,z,vx,vy,vz]

    Returns:
        array (3,3): matrix c
        array (3,3): matrix c_dot
    """
    r = rv_target[:3]
    v = rv_target[3:]
    n = np.cross(r, v)
    n_hat = n / np.linalg.norm(n)

    r_hat = r / np.linalg.norm(r)
    t_hat = np.cross(n_hat, r_hat)

    r_hat_dot = (v - np.dot(r_hat, v) * r_hat) / np.linalg.norm(r)

    n_hat_dot = np.array([0, 0, 0])
    t_hat_dot = np.cross(n_hat, r_hat_dot)

    c = np.vstack((r_hat, t_hat, n_hat))
    c_dot = np.vstack((r_hat_dot, t_hat_dot, n_hat_dot))

    return c, c_dot


def eci_to_rtn(rv_target, rv_chaser):
    """This function converts the ECI state vector of the chaser satellite to RTN coordinates centred\n
    on the target satellite

    Args:
        rv_target (array (6,1) or list ): state vector ECI of the target satellite [x,y,z,vx,vy,vz]
        rv_chaser (array (6,1) or list ): state vector ECI of the chaser satellite [x,y,z,vx,vy,vz]

    Returns:
        array (6,1) : state vector RTN of the chaser satellite
    """
    r_target = rv_target[:3]
    v_target = rv_target[3:]
    r_chaser = rv_chaser[:3]
    v_chaser = rv_chaser[3:]

    c, c_dot = compute_matrix_eci2rtn(rv_target)

    r_rel = r_chaser - r_target
    v_rel = v_chaser - v_target

    r_chaser_rtn = np.dot(c, r_rel)
    v_chaser_rtn = np.dot(c_dot, r_rel) + np.dot(c, v_rel)

    return np.concatenate((r_chaser_rtn, v_chaser_rtn))


def rtn_to_eci(rv_target, rv_chaser_rtn):
    """this function converts a state vector in RTN frame in state vector ECI

    Args:
        rv_target (array (6,1) or list ): state vector ECI of the target satellite [x,y,z,vx,vy,vz]
        rv_chaser_rtn (array (6,1) or list ): state vector RTN of the chaser satellite [x,y,z,vx,vy,vz]

    Returns:
        array (6,1) : state vector ECI of the chaser satellite
    """
    r_target_eci = rv_target[:3]
    v_target_eci = rv_target[3:]
    r_chaser_rtn = rv_chaser_rtn[:3]
    v_chaser_rtn = rv_chaser_rtn[3:]

    c, c_dot = compute_matrix_eci2rtn(rv_target)
    c = c.T
    c_dot = c_dot.T

    r_chaser_eci = np.dot(c, r_chaser_rtn) + r_target_eci
    v_chaser_eci = np.dot(c_dot, r_chaser_rtn) + np.dot(c, v_chaser_rtn) + v_target_eci

    return np.concatenate((r_chaser_eci, v_chaser_eci))


def sat_to_rv(satellite):
    """This function generates the state vector of a Poliastro satellite

    Args:
        satellite (Poliastro object)

    Returns:
        array (6,1): state vector ECI [x,y,z,vx,vy,vz]
    """
    return np.concatenate((satellite.r.value, satellite.v.value))
