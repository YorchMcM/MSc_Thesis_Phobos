'''
In this script we will define model A1. It includes:

· Rotational model: synchronous + once-per-orbit longitudinal libration of amplitude 1.1º (Rambaux et al. 2012)
· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: from spice.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
· Accelerations: Mars' harmonic coefficients up to degree and order 12. Phobos' quadrupole gravity field (C20 & C22).
· Cartesian state propagator.

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
body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
# Gravity field.
body_settings.get('Phobos').gravity_field_settings = let_there_be_a_gravitational_field('Phobos_body_fixed', 'QUAD', 'Le Maistre')
# And lastly the list of bodies is created.
bodies = environment_setup.create_system_of_bodies(body_settings)
# There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
bodies.get('Phobos').rotation_model.libration_calculator = environment.DirectLongitudeLibrationCalculator(np.radians(1.1))

bodies_to_propagate = ['Phobos']
central_bodies = ['Mars']

# ACCELERATION SETTINGS
acceleration_settings_on_phobos = dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)] )
acceleration_settings = { 'Phobos' : acceleration_settings_on_phobos }
acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)
# INTEGRATOR
time_step = 300.0  # These are 300s = 5min
coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                  coefficients,
                                                                                  time_step,
                                                                                  time_step,
                                                                                  np.inf, np.inf)
# PROPAGATION SETTINGS
# Initial conditions
initial_epoch = 0.0  # This is the J2000 epoch
initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)
# Termination condition
simulation_time = 90.0*constants.JULIAN_DAY
termination_condition = propagation_setup.propagator.time_termination(simulation_time)
# The settings object
propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                 acceleration_model,
                                                                 bodies_to_propagate,
                                                                 initial_state,
                                                                 initial_epoch,
                                                                 integrator_settings,
                                                                 termination_condition)
# SIMULATE DYNAMICS
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
save2txt(simulator.state_history, 'Pruebilla.txt')


