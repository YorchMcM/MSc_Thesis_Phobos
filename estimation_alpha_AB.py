import numpy as np
from time import time
from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import environment_setup, estimation_setup, estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

spice.load_standard_kernels()

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
trajectory_file = 'phobos-ephemerides-3500.txt'
imposed_trajectory = read_vector_history_from_file(trajectory_file)
phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source)
# There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies.get('Phobos').rotation_model.libration_calculator = environment.DirectLongitudeLibrationCalculator(scaled_amplitude)

# FIRST, WE PROPAGATE THE DYNAMICS AND WE SET PHOBOS' EPHEMERIS TO THE INTEGRATED RESULTS.
# WE ACTUALLY DON'T NEED THIS FOR THE POSITION OBSERVATIONS.

# PROPAGATOR
simulation_time = 3.0 * constants.JULIAN_YEAR
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, initial_epoch = 1.0 * constants.JULIAN_YEAR)

# PARAMETERS TO ESTIMATE
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Libration amplitude missing ???
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# LINK SET UP
# link_ends = { observation.LinkEndType.transmitter : observation.body_origin_link_end_id('Mars'),
#               observation.LinkEndType.receiver : observation.body_origin_link_end_id('Phobos') }
link_ends = { observation.LinkEndType.observed_body : observation.body_origin_link_end_id('Phobos') }
link = observation.link_definition(link_ends)
observation_settings = [observation.cartesian_position(link)]

# observation_simulators = estimation_setup.create_observation_simulators(observation_settings, bodies)  # What is this going to be used for?

# NOW, WE CREATE THE OBSERVATIONS

t0 = 3.5 * constants.JULIAN_YEAR
tf = 6.0 * constants.JULIAN_YEAR
dt = 1800.0
N = int((tf - t0) / dt) + 1
observation_times = np.linspace(t0, tf, N)

observable_type = estimation_setup.observation.ObservableType.position_observable_type
observation_simulation_settings = observation.tabulated_simulation_settings(observable_type,
                                                                            link,
                                                                            observation_times)

observation_model_settings = observation.cartesian_position(link)
observation_simulators = estimation_setup.create_observation_simulators([observation_model_settings], bodies)  # This is already a list!

observation_collection = estimation.simulate_observations([observation_simulation_settings],
                                                          observation_simulators,
                                                          bodies)

# AND NOW WE CREATE THE ESTIMATOR OBJECT, WE PROPAGATE THE VARIATIONAL EQUATIONS AND WE ESTIMATE
print('Going into the depths of tudat...')
tic = time()
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings,
    propagator_settings)
tac = time()
print('We\'re back! Variational equations propagated. Time taken:', (tac - tic) / 60.0, 'min')