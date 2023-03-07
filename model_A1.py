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
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
# libration_amplitude = 5.0  # In degrees
# ecc_scale = 0.015034568572550524
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_martian_system(phobos_ephemerides, gravity_field_type, gravity_field_source, scaled_amplitude)

# DEFINE PROPAGATION
simulation_time = 90.0*constants.JULIAN_DAY
dependent_variables_to_save = [ propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos'),  # 0, 1, 2
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars')]  # 6, 7, 8, 9, 10, 11
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time,
                                                       dependent_variables_to_save)

# SIMULATE DYNAMICS
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
save2txt(simulator.state_history, 'Pruebilla.txt')

# RETRIEVE & PLOT EULER ANGLES AND SUB-MARTIAN POINT
mars_mu = bodies.get('Mars').gravitational_parameter
epochs_array = np.array(list(simulator.state_history.keys()))
euler_history = result2array(extract_elements_from_history(simulator.dependent_variable_history, [0, 1, 2]))
euler_history[:,1:] = bring_inside_bounds(euler_history[:,1:] * (-1.0), 0.0, TWOPI)
submartian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [4, 5]))
submartian_point[:,1:] = bring_inside_bounds(submartian_point[:,1:], -PI, PI, include = 'upper')
keplerian_history = extract_elements_from_history(simulator.dependent_variable_history, [6, 7, 8, 9, 10, 11])
eccentricity = result2array(extract_elements_from_history(keplerian_history, 1))[:,1]
mean_motion = result2array(mean_motion_history_from_keplerian_history(keplerian_history, mars_mu))[:,1]


plt.figure()
plt.plot(epochs_array / 86400.0, euler_history[:,1] * 180.0 / np.pi, label = r'$\phi$')
plt.plot(epochs_array / 86400.0, euler_history[:,2] * 180.0 / np.pi, label = r'$\theta$')
plt.plot(epochs_array / 86400.0, euler_history[:,3] * 180.0 / np.pi, label = r'$\psi$')
plt.legend()
plt.grid()
plt.title('Euler angles')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Angle [º]')

plt.figure()
plt.plot(epochs_array / 86400.0, submartian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
plt.plot(epochs_array / 86400.0, submartian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
plt.legend()
plt.grid()
plt.title('Sub-martian point')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Coordinate [º]')

plt.figure()
plt.scatter(submartian_point[:,2] * 360.0 / TWOPI, submartian_point[:,1] * 360.0 / TWOPI)
plt.grid()
plt.ylim([-90.0, 90.0])
plt.xlim([-180.0, 180.0])
plt.title('Sub-martian point')
plt.xlabel('LON [º]')
plt.ylabel('LAT [º]')

# plt.figure()
# plt.plot(epochs_array / 86400.0, eccentricity)
# plt.grid()
# plt.title('Eccentricity')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$e$ [-]')

plt.figure()
plt.plot(epochs_array / 86400.0, mean_motion * 86400.0)
plt.grid()
plt.title('Mean motion')
plt.xlabel('Time [days since J2000]')
plt.ylabel(r'$n$ [rad/day]')

# print('Libration amplitude: ' + str(libration_amplitude))
# print('Pre propagation eccentricity: ' + str(ecc_scale))
# print('Post propagation eccentricity: ' + str(np.mean(eccentricity)))
# print('Increment: ' + str(np.mean(eccentricity) - ecc_scale))
# print('Absolute change: ' + str(abs(np.mean(eccentricity) - ecc_scale)))