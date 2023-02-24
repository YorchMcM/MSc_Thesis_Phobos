'''
In this script we will define model A2. It includes:

· Translational model: the states output by model A1
· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: damped initial state provided by Tudat.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
· Torques: Mars' center of mass on Phobos' quadrupole gravity field.
· Propagator: Quaternion and angular velocity vector.

'''

# IMPORTS
import numpy as np
from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
# from time import time
# from os import getcwd
from Auxiliaries import *

from tudatpy.kernel.interface import spice
# from tudatpy.kernel import numerical_simulation
# from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, estimation_setup
from tudatpy.kernel import constants
# from tudatpy.util import result2array
# from tudatpy.io import save2txt
# from tudatpy.plotting import trajectory_3d

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
numerical_states = read_vector_history_from_file('Pruebilla.txt')
phobos_ephemerides = environment_setup.ephemeris.tabulated(numerical_states, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_martian_system(phobos_ephemerides, gravity_field_type, gravity_field_source)
# There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
bodies.get('Phobos').inertia_tensor = inertia_tensor_from_spherical_harmonic_gravity_field(
    bodies.get('Phobos').gravity_field_model
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
simulation_time = 270.0*constants.JULIAN_DAY
termination_condition = propagation_setup.propagator.time_termination(simulation_time)
# The settings object
propagator_settings = propagation_setup.propagator.rotational(torque_model,
                                                              bodies_to_propagate,
                                                              initial_state,
                                                              initial_epoch,
                                                              integrator_settings,
                                                              termination_condition)

# Now that we have all integration and propagation settings, we compute the undamped initial rotational state.
# phobos_mean_rotational_rate = 19.694 / constants.JULIAN_DAY  # In rad/day
phobos_mean_rotational_rate = 19.6954 / constants.JULIAN_DAY  # In rad/day
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0])*3600.0)  # In seconds.
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)

damped_initial_state = damping_results.initial_state
# Now that we have the damped initial state, we propagate the rotational dynamics.
propagator_settings = propagation_setup.propagator.rotational(torque_model,
                                                              bodies_to_propagate,
                                                              damped_initial_state,
                                                              initial_epoch,
                                                              integrator_settings,
                                                              termination_condition)
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

# POST-PROCESS
mars_mu = bodies.get('Mars').gravitational_parameter
mean_motion_history = cartesian_to_keplerian_mean_history(numerical_states, mars_mu)
mean_motion_history = extract_element_from_history(mean_motion_history, [0])
mean_motion_history = semi_major_axis_to_mean_motion_history(mean_motion_history, mars_mu)
mean_motion_history = result2array(mean_motion_history)
libration_history = result2array(get_libration_history(numerical_states, simulator.state_history, mars_mu))
epochs_array = libration_history[:,0]

# print('Average mean motion: ' + str(np.mean(mean_motion_history[:,1]) * 86400.0) + ' rad/day.')

# plt.figure()
# plt.plot(epochs_array / 86400.0, euler_angle_history[:,1] * 180.0 / np.pi, label = r'$\psi$')
# plt.plot(epochs_array / 86400.0, euler_angle_history[:,2] * 180.0 / np.pi, label = r'$\theta$')
# plt.plot(epochs_array / 86400.0, euler_angle_history[:,3] * 180.0 / np.pi, label = r'$\phi$')
# plt.legend()
# plt.grid()
# plt.title('Euler angles')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel('Angle [º]')

# plt.figure()
# plt.plot(epochs_array / 86400.0, mean_anomaly_history[:,1] * 180.0 / np.pi)
# plt.grid()
# plt.title('Mean anomaly')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$M$ [º]')

# plt.figure()
# plt.plot(epochs_array / 86400.0, mean_motion_history[:,1] * 86400.0)
# plt.grid()
# plt.title('Mean motion')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$n$ [rad/day]')

plt.figure()
plt.plot(epochs_array / 86400.0, libration_history[:,1] * 180.0 / np.pi)
plt.grid()
plt.title(r'Libration angle ($\omega_o$ = ' + str(phobos_mean_rotational_rate * constants.JULIAN_DAY) + ' rad/day)')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Angle [º]')