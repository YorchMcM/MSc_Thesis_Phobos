'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
    · Reference radius: 13 km
'''

import numpy as np
from Auxiliaries import get_solar_system, get_propagator_settings, mat2quat, inertial_to_rsw_rotation_matrix
from tudatpy.kernel.interface.spice import get_body_cartesian_state_at_epoch
from tudatpy.kernel import constants

bodies = get_solar_system('C', 'ephemeris/translation-a.eph')
phobos = bodies.get('Phobos')

initial_state = get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'None', 0.0)
# initial_state = mat2quat(inertial_to_rsw_rotation_matrix(initial_state))
initial_state = np.concatenate((initial_state, mat2quat(inertial_to_rsw_rotation_matrix(initial_state))), 0)
initial_state = np.concatenate((initial_state, np.array([0, 0, 1])), 0)
propagator = get_propagator_settings('C', bodies, 0.0, initial_state, 30.0*constants.JULIAN_DAY)