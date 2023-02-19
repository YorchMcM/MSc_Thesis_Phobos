'''
In this script we will define model A2. It includes:

· Translational model: the states output by model A1
· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: damped initial state provided by Tudat.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
· Torques: Mars' center of mass on Phobos' quadrupole gravity field.
· Quaternion/angular velocity propagator.

'''

# IMPORTS
import numpy as np
from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from time import time
from os import getcwd
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, estimation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.io import save2txt
from tudatpy.plotting import trajectory_3d
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
from tudatpy.kernel.astro.element_conversion import quaternion_entries_to_rotation_matrix as quat2mat

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'C:\Users\Yorch\OneDrive - Delft University of Technology\Year 2022-2023\MSc_Thesis_Phobos\Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

# LOAD SPICE KERNELS
spice.load_standard_kernels()

# CREATE YOUR UNIVERSE (OR AT LEAST MARS, FOR WHAT I USE DEFAULTS FROM SPICE)
bodies_to_create = ["Mars"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# BUILT-IN INFORMATION ON PHOBOS IS QUITE CRAP. WE WILL REMAKE THE WHOLE BODY OF PHOBOS OURSELVES BASED ON LE MAISTRE (2019).
body_settings.add_empty_settings('Phobos')
# Ephemeris and rotation models.
body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
numerical_states = read_vector_history_from_file('Pruebilla.txt')
body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.tabulated(numerical_states, 'Mars', 'J2000')
# Gravity field.
body_settings.get('Phobos').gravity_field_settings = let_there_be_a_gravitational_field('Phobos_body_fixed', 'QUAD', 'Le Maistre')
# And lastly the list of bodies is created.
bodies = environment_setup.create_system_of_bodies(body_settings)
# There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
bodies.get('Phobos').inertia_tensor = inertia_tensor_from_spherical_harmonic_gravity_field_settings(
    body_settings.get('Phobos').gravity_field_settings
)

bodies_to_propagate = ['Phobos']
central_bodies = ['Mars']

# TORQUE SETTINGS
torque_settings_on_phobos = dict( Mars = [propagation_setup.torque.spherical_harmonic_gravitational(2, 2)] )
torque_settings = { 'Phobos' : torque_settings_on_phobos }
torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)
# INTEGRATOR
time_step = 300.0  # These are 300s = 5min
coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
# Esta linea se rompe. Dice que incompatible arguments.
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                  coefficients,
                                                                                  time_step,
                                                                                  time_step,
                                                                                  np.inf, np.inf)
# PROPAGATION SETTINGS
# Initial conditions. This initial state is generic and "fake". Computation of the undamped state requires the use
# of the get_zero_mode_rotational_state function, which takes a PropagatorSettings object as input. However, this
# PropagatorSettings object takes an initial condition as argument. I assume this circularity is resolved by creating
# a PropagatorSettings object with an arbitrary initial condition that will be used as initial guess for the
# get_zero_proper_mode_rotational_state function.
initial_epoch = 0.0  # This is the J2000 epoch
initial_rotation_matrix = np.eye(3)
initial_angular_velocity = np.zeros(3)
initial_state = np.zeros(7)
initial_state[:4] = mat2quat(initial_rotation_matrix)
initial_state[4:] = initial_angular_velocity
# Termination condition
simulation_time = 90.0*constants.JULIAN_DAY
termination_condition = propagation_setup.propagator.time_termination(simulation_time)
# The settings object
print('pre')
propagator_settings = propagation_setup.propagator.rotational(torque_model,
                                                              bodies_to_propagate,
                                                              initial_state,
                                                              initial_epoch,
                                                              integrator_settings,
                                                              termination_condition)
print('post')

# Now that we have all integration and propagation settings, we compute the undamped initial rotational state.
# A preliminary documentation of the Python-exposed get_zero_proper_mode_rotational_state function has been redacted
# in the complementary_documentation.txt file.
phobos_mean_rotational_rate = 19.694 / constants.JULIAN_DAY  # In rad/s
dissipation_times = list(np.array([10, 20, 30, 40])*constants.JULIAN_DAY)  # In seconds.
damped_initial_state, damped_states = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                                             integrator_settings,
                                                                                                             propagator_settings,
                                                                                                             phobos_mean_rotational_rate,
                                                                                                             dissipation_times)