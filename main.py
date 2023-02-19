# IMPORTS
import numpy as np
from matplotlib import pyplot as plt
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, estimation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array

# models = [False, False, False, False]  # This will automatically exit the program.
models = [True, False, False, False]  # This will run model A1.
# models = [False, True, False, False]  # This will run model A2.
# models = [False, False, True, False]  # This will run model B.
# models = [False, False, False, True]  # This will run model C.

if not any(models):
    print('No models to run. Terminating program.')
    exit()

# LOAD SPICE KERNELS
spice.load_standard_kernels()

# CREATE YOUR UNIVERSE (OR AT LEAST MARS, FOR WHAT I USE DEFAULTS FROM SPICE)
bodies_to_create = ["Mars"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)


# BUILT-IN INFORMATION ON PHOBOS IS QUITE CRAP. WE WILL REMAKE THE WHOLE BODY OF PHOBOS OURSELVES BASED ON SCHEERES (2019). THE PHOBOS WE WILL CREATE DEPENDS ON THE MODEL TO BE RUN.
body_settings.add_empty_settings('Phobos')


# SETTINGS A1. Here, we need a rotation model.
if models[0]:
    body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars',
                                                                                                       'J2000',
                                                                                                       'Phobos_body_fixed')
    body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')  # Apparently if Phobos doesn't have an ephemeris setting tudat breaks.


# SETTINGS A2. Here, we need an ephemeris model.
if models[1]:
    history_a1 = None  # Here, history_a1 will be the history that we get when integrating model A1. If it does not exist yet, it will be assigned to be the spice default.
    try:
        body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.tabulated(history_a1,
                                                                                               frame_origin = 'Mars',
                                                                                               frame_orientation = 'J2000')
    except:
        body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')  # Apparently if Phobos doesn't have an ephemeris setting tudat breaks.

    body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.spice('J2000', 'Phobos_body_fixed')  # For completeness.


# SETTINGS A/B. Here we need a quadrupole gravitational field for Phobos.
if models[0] or models[1] or models[2]:
    body_settings.get('Phobos').gravity_field_settings = let_there_be_a_gravitational_field('Phobos_body_fixed', 'QUAD')


# SETTINGS C. Here we need the "full" gravitational field for Phobos (up to degree and order 4).
if models[3]:
    body_settings.get('Phobos').gravity_field_settings = let_there_be_a_gravitational_field('Phobos_body_fixed', 'FULL')

bodies = environment_setup.create_system_of_bodies(body_settings)


# SETTINGS A2/B/C. In any of these cases, we will need an inertia tensor. It will be computed using Phobos' gravitational field.
if models[1] or models[2] or models[3]:
    bodies.get('Phobos').inertia_tensor = inertia_tensor_from_spherical_harmonic_gravity_field_settings(body_settings.get('Phobos').gravity_field_settings)


# NOW THAT PHOBOS EXISTS AND HAS ALL PROPERTIES WE NEED IT TO HAVE, WE START WITH THE DYNAMICS THEMSELVES.

bodies_to_propagate = ['Phobos']
central_bodies = ['Mars']

# MODEL A1
if models[0]:
    acceleration_settings_on_phobos = dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)] )

# MODEL A2
if models[1]:
    torque_settings_on_phobos = dict( Phobos = [propagation_setup.torque.spherical_harmonic_gravitational(2, 2)] )

# MODEL B
if models[2]:
    acceleration_settings_on_phobos = dict( Mars=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)])
    torque_settings_on_phobos = dict( Phobos=[propagation_setup.torque.spherical_harmonic_gravitational(2, 2)] )

# MODEL C
if models[3]:
    acceleration_settings_on_phobos = dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 4, 4)] )
    torque_settings_on_phobos = dict( Phobos = [propagation_setup.torque.spherical_harmonic_gravitational(4, 4)])


if models[0] or models[2] or models[3]:
    acceleration_settings = { 'Phobos' : acceleration_settings_on_phobos }
    acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)

if models[1] or models[2] or models[3]:
    torque_settings = { 'Phobos' : torque_settings_on_phobos }
    torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)


# Initial and termination times. We start at 01/01/2000 00:00 (or 12:00?). For tests, we will integrate for 90 days.
initial_epoch = constants.JULIAN_DAY_ON_J2000
final_epoch = (constants.JULIAN_DAY_ON_J2000 + 90.0)*constants.JULIAN_DAY
# Initial conditions
initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)

# # Initial conditions.
# if models[0]:
#     initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)
#
# if models[1]: pass
# if models[2]: pass
# if models[3]: pass
#
# # We select the integrator. We'll play around with this thing.
# integrators = [ propagation_setup.integrator.runge_kutta_4,
#                 propagation_setup.integrator.runge_kutta_variable_step_size,
#                 propagation_setup.integrator.adams_bashforth_moulton ]
# runge_kutta_coefficient_sets = [ propagation_setup.integrator.rkf_78,
#                                  propagation_setup.integrator.rkf_89,
#                                  propagation_setup.integrator.rkf_108 ]
# runge_kutta_time_steps = [ 1.0, 2.0, 5.0, 10.0, 30.0, 60.0 ]
# integrator_settings_list = [None]*(len(runge_kutta_time_steps)*(2+len(runge_kutta_coefficient_sets)))
# file_names = [None]*(len(runge_kutta_time_steps)*(2+len(runge_kutta_coefficient_sets)))
#
# idx = -1
# for current_integrator in integrators:
#
#     if current_integrator is propagation_setup.integrator.runge_kutta_4:
#         integrator_string = 'rk4'
#         for current_time_step in runge_kutta_time_steps:
#             idx = idx+1
#             time_step_string = str(current_time_step)
#             integrator_settings_list[idx] = current_integrator(current_time_step)
#             file_names[idx] = integrator_string + '_' + time_step_string
#
#     if current_integrator is propagation_setup.integrator.runge_kutta_variable_step_size:
#         integrator_string = 'runge_kutta'
#         for current_coefficient_set in runge_kutta_coefficient_sets:
#             # coefficient_string =
#             for current_time_step in runge_kutta_time_steps:
#                 idx = idx+1


# integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
#     initial_time, time_step, coefficient_set,
#     time_step, time_step,
#     np.inf, np.inf)
