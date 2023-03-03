'''
In this script we will define model B. It includes:

· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial translational state: from spice.
· Initial rotational state: damped initial state provided by Tudat.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 3 minutes
· Accelerations: Mars' harmonic coefficients up to degree and order 12. Phobos' quadrupole gravity field (C20 & C22).
· Torques: Mars' center of mass on Phobos' quadrupole gravity field.
· Translational propagator: Cartesian states
· Rotational propagator: Quaternion and angular velocity vector.

'''

# IMPORTS
import numpy as np
from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
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

print('Creating Martian system...')
# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
# The ephemeris model is irrelevant because the translational dynamics of Phobos will be propagated.
phobos_ephemerides = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_martian_system(phobos_ephemerides, gravity_field_type, gravity_field_source)
# There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
bodies.get('Phobos').inertia_tensor = inertia_tensor_from_spherical_harmonic_gravity_field(
    bodies.get('Phobos').gravity_field_model
)

bodies_to_propagate = ['Phobos']
central_bodies = ['Mars']

print('Creating accelerations...')
# ACCELERATION SETTINGS
acceleration_settings_on_phobos = dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)] )
acceleration_settings = { 'Phobos' : acceleration_settings_on_phobos }
acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)

print('Creating torques...')
# TORQUE SETTINGS
torque_settings_on_phobos = dict( Mars = [propagation_setup.torque.spherical_harmonic_gravitational(2, 2)] )
torque_settings = { 'Phobos' : torque_settings_on_phobos }
torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

print('Building integrator...')
# INTEGRATOR
time_step = 180.0  # These are 180s = 3min
coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                  coefficients,
                                                                                  time_step,
                                                                                  time_step,
                                                                                  np.inf, np.inf)

print('Defining initial and termination conditions...')
# INITIAL CONDITIONS
# Initial epoch
initial_epoch = 0.0  # This is the J2000 epoch
# Initial translational state
initial_state_trans = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)
# Initial rotational state
initial_rotation_matrix = np.eye(3)
initial_angular_velocity = np.zeros(3)
initial_state_rot = np.zeros(7)
initial_state_rot[:4] = mat2quat(initial_rotation_matrix)
initial_state_rot[4:] = initial_angular_velocity
# Termination condition
simulation_time = 90.0*constants.JULIAN_DAY
termination_condition = propagation_setup.propagator.time_termination(simulation_time)

# PROPAGATORS
print('Building translational propagator...')
# Translational propagator
propagator_settings_trans = propagation_setup.propagator.translational(central_bodies,
                                                                       acceleration_model,
                                                                       bodies_to_propagate,
                                                                       initial_state_trans,
                                                                       initial_epoch,
                                                                       integrator_settings,
                                                                       termination_condition)
print('Building rotational propagator...')
# Rotational propagator
propagator_settings_rot = propagation_setup.propagator.rotational(torque_model,
                                                                  bodies_to_propagate,
                                                                  initial_state_rot,
                                                                  initial_epoch,
                                                                  integrator_settings,
                                                                  termination_condition)
# Multi-type propagator
print('Putting propagators together...')
propagator_list = [propagator_settings_trans, propagator_settings_rot]
propagator_settings = propagation_setup.propagator.multitype(propagator_list,
                                                             integrator_settings,
                                                             initial_epoch,
                                                             termination_condition)

# COMPUTATION OF DAMPED INITIAL STATE
print('Computing damped dynamics...')
phobos_mean_rotational_rate = 19.694 / constants.JULIAN_DAY  # In rad/s
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0])*3600.0)  # In seconds.
print('pre')
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)
print('post')
damped_initial_state = damping_results.initial_state

# SIMULATION OF DAMPED DYNAMICS
propagator_settings = propagation_setup.propagator.rotational(torque_model,
                                                              bodies_to_propagate,
                                                              damped_initial_state,
                                                              initial_epoch,
                                                              integrator_settings,
                                                              termination_condition)
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)