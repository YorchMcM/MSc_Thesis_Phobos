import numpy as np
from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import environment_setup, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

spice.load_standard_kernels()

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
phobos_ephemerides = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_martian_system(phobos_ephemerides, gravity_field_type, gravity_field_source)
# There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
libration_amplitude = 1.1  # In degrees
libration_amplitude = np.radians(libration_amplitude)  # In radians
bodies.get('Phobos').rotation_model.libration_calculator = environment.DirectLongitudeLibrationCalculator(libration_amplitude)

# PROPAGATOR
simulation_time = 90.0 * constants.JULIAN_DAY
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time)

# PARAMETERS TO ESTIMATE
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Libration amplitude missing ???
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# LINK SET UP
link_ends = { observation.LinkEndType.transmitter : observation.body_origin_link_end_id('Mars'),
              observation.LinkEndType.receiver : observation.body_origin_link_end_id('Phobos') }
link = observation.link_definition(link_ends)
observation_settings = [observation.cartesian_position(link)]
print('0')
# observation_simulators = estimation_setup.create_observation_simulators(observation_settings, bodies)  # What is this going to be used for?

print('pre')
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings,
    propagator_settings)
print('post')