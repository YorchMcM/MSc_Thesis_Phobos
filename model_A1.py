'''
In this script we will define model A1. It includes:

· Rotational model: synchronous + once-per-orbit longitudinal libration of amplitude 1.1º (Rambaux et al. 2012)
· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: from spice.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
· Accelerations: Mars' harmonic coefficients up to degree and order 12. Phobos' quadrupole gravity field (C20 & C22).
· Propagator: Cartesian states

'''

# IMPORTS
import numpy as np
from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.io import save2txt

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'C:\Users\Yorch\OneDrive - Delft University of Technology\Year 2022-2023\MSc_Thesis_Phobos\Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

# LOAD SPICE KERNELS
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

simulation_time = 90.0*constants.JULIAN_DAY
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time)

# bodies_to_propagate = ['Phobos']
# central_bodies = ['Mars']
#
# # ACCELERATION SETTINGS
# acceleration_settings_on_phobos = dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)] )
# acceleration_settings = { 'Phobos' : acceleration_settings_on_phobos }
# acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)
# # INTEGRATOR
# time_step = 300.0  # These are 300s = 5min
# coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
# integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
#                                                                                   coefficients,
#                                                                                   time_step,
#                                                                                   time_step,
#                                                                                   np.inf, np.inf)
# # PROPAGATION SETTINGS
# # Initial conditions
# initial_epoch = 0.0  # This is the J2000 epoch
# initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)
# # Termination condition
# simulation_time = 90.0*constants.JULIAN_DAY
# termination_condition = propagation_setup.propagator.time_termination(simulation_time)
# # The settings object
# propagator_settings = propagation_setup.propagator.translational(central_bodies,
#                                                                  acceleration_model,
#                                                                  bodies_to_propagate,
#                                                                  initial_state,
#                                                                  initial_epoch,
#                                                                  integrator_settings,
#                                                                  termination_condition)
# SIMULATE DYNAMICS
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
save2txt(simulator.state_history, 'Pruebilla.txt')

# RETRIEVE LIBRATION HISTORY
states = read_vector_history_from_file('Pruebilla.txt')
mars_mu = bodies.get('Mars').gravitational_parameter
libration_history = result2array(get_longitudinal_libration_history_from_libration_calculator(states,
                                                                                              mars_mu, libration_amplitude))
plt.figure()
plt.plot(libration_history[:,0] / 86400.0, libration_history[:,1] * 360 / TWOPI)
plt.title(r'Libration angle')
plt.ylabel(r'$\theta$ [º]')
plt.xlabel(r'Time [days since J2000]')
plt.grid()





# print('FINISHED RUNNING MODEL A1.')