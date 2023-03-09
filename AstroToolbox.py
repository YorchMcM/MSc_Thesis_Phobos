import numpy as np
from numpy import array, cos, sin, tan, arctan2, arccos, arcsin, sqrt, cross, dot, degrees, radians
from numpy import pi as PI
from numpy.linalg import norm
from scipy.optimize import fsolve
TWOPI = 2*PI

# def radians(d, m = 0.0, s = 0.0): return np.radians(d + m/60.0 + s/3600.0)


def a2n(a, mu = 398600.0): return sqrt(mu/a**3)


def non_linear_eq(x, *args): return x-args[0]*sin(x)-args[1]


def M2E(M, e): return fsolve(non_linear_eq, M, args=(e, M))[0]


def E2theta(E, e):
    den = 1 - e*cos(E)
    sine = (sqrt(1-e**2)*sin(E))/den
    cosine = (cos(E) - e)/den
    return arctan2(sine, cosine)


def M2theta(M, e):
    theta = E2theta(M2E(M, e), e)
    while True:
        if theta < 0.0 : theta = theta + TWOPI
        else: break
    while True:
        if theta > TWOPI: theta = theta - TWOPI
        else: break
    return theta


def theta2E(theta, e):

    '''
    Will return a value between 0 and 2π, no matter the value of θ
    '''

    den = 1+e*cos(theta)
    sine = (sqrt(1-e**2)*sin(theta))/den
    cosine = (e+cos(theta))/den
    E = np.arctan2(sine, cosine)
    if E < 0.0 : E = E + TWOPI
    return E


def E2M(E, e): return E-e*sin(E) # This is between 0 and 2π


def theta2M(theta, e, allowForWaitRevolutions = False):
    M = E2M(theta2E(theta, e), e) # This is between 0 and 2π

    if allowForWaitRevolutions:
        N = theta // TWOPI
        if theta % TWOPI == 0.0 and N > 0: N = N-1
        M = M + N*TWOPI

    return M


def increment_theta(theta_o, dt, e, n):

    '''
    WARNING : The true anomaly must be passed in RADIANS.
    '''

    print('WARNING: (AstroToolbox --> increment_theta) The argument "theta_o" must be passed in RADIANS. Make sure this is the case. The output is also in radians.')
    M_o = theta2M(theta_o, e)
    M = M_o + n*dt
    theta = M2theta(M, e)

    while True:
        if theta < 0.0: theta = theta + TWOPI
        else: break

    while True:
        if theta > TWOPI: theta = theta - TWOPI
        else: break

    return theta


def BCI2COE(r_v: array(3), v_v: array(3), mu = 398600.441):

    '''

    Converts the Body-Centered Inertial state vector (position and velocity) in Cartesian coordinates to the Classical
    Orbital Elements state. All angles are between 0º and 360º. The inclination angle is only defined between 0º and 180º.

    NOTE: Units can be freely chosen, but must be consistent. If input position, velocity and μ ar in meters, the
    resulting semi-major axis will also be in meter. If they are in km, so will the output.

    INPUTS:
    · r_v : (1D array with 3 entries) Position vector. [km , m]
    · v_v : (1D array with 3 entries) Velocity vector. [km/s, m/s]
    · mu : (Scalar, optional) The gravitational parameter of the central body (default is Earth). [km^3/s^3, m^3/s^2]

    OUTPUTS:
    · a : (Scalar) Semi-major axis. [km, m]
    · e : (Scalar) Eccentricity. [-]
    · i : (Scalar) Inclination. [º]
    · RAAN : (Scalar) Right ascension of ascending node. [º]
    · omega : (Scalar) Argument of periapsis. [º]
    · theta : (Scalar) True anomaly. [º]

    '''

    # INERTIAL FRAME
    X = array([1, 0, 0])
    Y = array([0, 1, 0])
    Z = array([0, 0, 1])

    # AUXILIARY COMPUTATIONS
    r = norm(r_v)
    r_hat = r_v/r
    v = norm(v_v)
    # v_hat = v_v/v - Useless. Only here for completeness
    h_v = cross(r_v,v_v)
    h = norm(h_v)
    e_v = cross(v_v,h_v)/mu - r_v/r
    e = norm(e_v)                           ### RESULT
    a = -mu/2/(v*v/2-mu/r)                  ### RESULT

    # PERIFOCAL FRAME
    p = e_v/e
    w = h_v/h
    q = cross(w,p)

    # LINE OF NODES
    n = cross(Z,w)
    n = n/norm(n)

    # ANGLES
    # Inclination
    cosine = dot(w,Z)
    i = degrees(arccos(cosine))             ### RESULT

    # Right ascension of ascending node
    cosine = dot(n,X)
    sine = dot(n,Y)
    RAAN = degrees(arctan2(sine, cosine))   ### RESULT
    if RAAN < 0 : RAAN = RAAN + 360.0

    # Argument of periapsis
    aux = cross(w,n)
    cosine = dot(p,n)
    sine = dot(p,aux)
    omega = degrees(arctan2(sine, cosine))  ### RESULT
    if omega < 0 : omega = omega + 360.0

    # True anomaly
    cosine = dot(r_hat,p)
    sine = dot(r_hat,q)
    theta = degrees(arctan2(sine, cosine))  ### RESULT
    if theta < 0 : theta = theta + 360.0

    return array([a, e, i, RAAN, omega, theta])


def COE2BCI(COE, mu = 398600.441):

    '''

    Converts the Classical Orbital Elements state into Body-Centered Inertial state vector (position and velocity) in
    Cartesian coordinates. All angles must are between 0º and 360º. The inclination angle is only defined between 0º and 180º.

    NOTE: Units can be freely chosen, but must be consistent. If input semi-major axis and μ are in meters, the resulting
    position and velocity will be in m and m/s respectevely; if the inputs are in km, so will the outputs.

    INPUTS: (Inside the 1D array "COE", in this order)
    · a : (Scalar) Semi-major axis. [km]
    · e : (Scalar) Eccentricity. [-]
    · i : (Scalar) Inclination. [º]
    · RAAN : (Scalar) Right ascension of ascending node. [º]
    · omega : (Scalar) Argument of periapsis. [º]
    · theta : (Scalar) True anomaly. [º]
    · mu : (Scalar, optional) The gravitational parameter of the central body (default is Earth). [km^3/s^3, m^3/s^2]

    OUTPUTS:
    · r_v : (1D array with 3 entries) Position vector. [km]
    · v_v : (1D array with 3 entries) Velocity vector. [km/s]

    '''

    a, e, i, RAAN, omega, theta = COE

    i = radians(i)
    RAAN = radians(RAAN)
    omega = radians(omega)
    theta = radians(theta)

    # SCALAR QUANTITIES
    mu_h = sqrt(mu/a/(1-e**2)) # Gravitational parameter over angular momentum
    r = a*(1-e**2)/(1+e*cos(theta)) # Distance from central body
    v_r = mu_h*e*sin(theta) # Radial velocity
    v_theta = mu_h*(1+e*cos(theta)) # Tangential velocity

    # STATE VECTOR (POSITION AND VELOCITY) IN RTN FRAME.
    r_RTN = array([r, 0.0, 0.0])
    v_RTN = array([v_r, v_theta, 0.0])


    # ROTATION MATRICES
    RTN_R_P = array([[cos(theta), sin(theta), 0.0], [-sin(theta), cos(theta), 0.0], [0.0, 0.0, 1.0]])   # Perifocal to RTN
    P_R_RTN = RTN_R_P.T                                                                                 # RTN to Perifocal

    A_R_I = array([[cos(RAAN), sin(RAAN), 0.0], [-sin(RAAN), cos(RAAN), 0.0], [0.0, 0.0, 1.0]])
    B_R_A = array([[1.0, 0.0, 0.0], [0.0, cos(i), sin(i)], [0.0, -sin(i), cos(i)]])
    P_R_B = array([[cos(omega), sin(omega), 0.0], [-sin(omega), cos(omega), 0.0], [0.0, 0.0, 1.0]])

    P_R_I = P_R_B @ B_R_A @ A_R_I   # Inertial to Perifocal
    I_R_P = P_R_I.T                 # Perifocal to Inertial

    I_R_RTN = I_R_P @ P_R_RTN # RTN to Inertial

    # STATE VECTOR (POSITION AND VELOCITY) IN INERTIAL FRAME
    r_I = I_R_RTN @ r_RTN
    v_I = I_R_RTN @ v_RTN

    return array([r_I, v_I]).reshape(6)


def UTC2JD(date: str, time: str = None):
    '''
    From:
    "Fundamentals of Astrodynamics" by Karel F. Wakker. First equation on page 260.
    OR
    "Orbital Mechanics for Engineering Students" by Howard D. Curtis, 3rd ed. Equation 5.48 on page 259.
    Note: The Julian date starts at NOON. Calendar date 01/01/2000 at 00:00:00 (midnight) corresponds to JD = 2451544.5,
    while JD = 2451545 is the Julian day of date 01/01/2000 at 12:00:00 (noon).
    '''
    day, month, year = date.split('/')
    JD = 1721013.5 + 367.0*int(year) - int((7/4)*(int(year)+int((int(month)+9)/12)))+int((275/9)*int(month))+int(day)
    if time is not None:
        hour, minute, second = time.split(':')
        JD = JD+(int(hour)+int(minute)/60.0+float(second)/3600.0)/24
    return JD


def cartesian_to_rtn_rotation_matrix(state: array(6)) -> array([3,3]):

    # COMPUTE RTN VECTORS
    r = state[:3]
    v = state[3:]

    r_hat = r / norm(r)          # R
    h = cross(r, v)
    h_hat = h / norm(h)          # N
    t_hat = cross(h_hat, r_hat)  # T

    # BUILD ROTATION MATRIX
    R = np.zeros([3, 3])
    R[:,0] = r_hat
    R[:,1] = t_hat
    R[:,2] = h_hat

    return R


def get_synchronous_rotation_matrix_to_base_frame(state: array(6)) -> array([3,3]):

    base_to_rtn_matrix = cartesian_to_rtn_rotation_matrix(state)
    base_to_rtn_matrix[:,:2] = base_to_rtn_matrix[:,:2] * (-1.0)
    synchronous_to_base_matrix = base_to_rtn_matrix.T

    return synchronous_to_base_matrix

