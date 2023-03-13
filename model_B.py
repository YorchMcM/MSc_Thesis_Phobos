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
# import sys
# import numpy as np
from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *

sys.path.insert(0, '/home/yorch/tudat-bundle/cmake-build-release/tudatpy')

from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
from tudatpy.io import save2txt

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

# LOAD SPICE KERNELS
spice.load_standard_kernels()

print('Creating Martian system...')
# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
# The ephemeris model is irrelevant because the translational dynamics of Phobos will be propagated. But tudat complains if Phobos doesn't have one.
phobos_ephemerides = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_martian_system(phobos_ephemerides, gravity_field_type, gravity_field_source)

# PHOBOS' ROTATIONAL PROPERTIES
# phobos_mean_rotational_rate = 0.00022785759213999574  # In rad/s
phobos_mean_rotational_rate = 0.000227995  # In rad/s
phobos_rotational_speed_at_periapsis = phobos_mean_rotational_rate * (1.0 + np.radians(0.0))

# DEFINE PROPAGATION
initial_epoch = 13035.0  # This is (approximately) the first periapsis passage since J2000
simulation_time = 2.0 * constants.JULIAN_DAY
# initial_rotational_speed = 0.00022785759213999574  # In rad/s
dependent_variables_to_save = [ propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos'),  # 0, 1, 2
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars') ]  # 6, 7, 8, 9, 10, 11

# SIMULATE UNDAMPED DYNAMICS FOR REFERENCE
print('Simulating undamped dynamics...')
fake_initial_translational_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)
fake_initial_rotational_state = get_fake_initial_state(bodies, initial_epoch, phobos_rotational_speed_at_periapsis)
fake_initial_state = np.concatenate((fake_initial_translational_state, fake_initial_rotational_state))
fake_propagator_settings = get_model_b_propagator_settings(bodies, simulation_time, fake_initial_state, dependent_variables_to_save)
simulator_undamped = numerical_simulation.create_dynamics_simulator(bodies, fake_propagator_settings)

# OBTAIN DAMPED INITIAL STATE
# dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0])*3600.0)  # In seconds.
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0])*3600.0)  # In seconds.
percentages = np.linspace(0.0, 0.1, 201)
print('Computing damped initial condition...')
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         fake_propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)
damped_initial_state = damping_results.initial_state

# SIMULATE DAMPED DYNAMICS
propagator_settings = get_model_b_propagator_settings(bodies, simulation_time, damped_initial_state, dependent_variables_to_save)
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

# POST-PROCESS
epochs_array = np.array(list(simulator.state_history.keys()))
clean_signal = [TWOPI, 1]
# Undamped dynamics
keplerian_history = extract_elements_from_history(simulator_undamped.dependent_variable_history, [6, 7, 8, 9, 10, 11])
euler_history_undamped = extract_elements_from_history(simulator_undamped.dependent_variable_history, [2, 1, 0])
euler_history_undamped = result2array(euler_history_undamped)
euler_history_undamped[:,1:] = -euler_history_undamped[:,1:]
euler_history_undamped = array2result(euler_history_undamped)
psi_freq_undamped, psi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 0), clean_signal)
theta_freq_undamped, theta_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 1), clean_signal)
phi_freq_undamped, phi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 2), clean_signal)
euler_history_undamped = result2array(euler_history_undamped)
euler_history_undamped[:,1:] = bring_inside_bounds(euler_history_undamped[:,1:], 0.0, TWOPI)

submartian_point_undamped = result2array(extract_elements_from_history(simulator_undamped.dependent_variable_history, [4, 5]))
submartian_point_undamped[:,1:] = bring_inside_bounds(submartian_point_undamped[:,1:], -PI, PI, include = 'upper')

# undamped_theta = result2array(extract_elements_from_history(simulator_undamped.dependent_variable_history, 11))
# undamped_theta[:,1] = bring_inside_bounds(undamped_theta[:,1], 0.0, TWOPI)

# Damped dynamics
euler_history = extract_elements_from_history(simulator.dependent_variable_history, [2, 1, 0])
euler_history = result2array(euler_history)
euler_history[:,1:] = -euler_history[:,1:]
euler_history = array2result(euler_history)
psi_freq, psi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 0), clean_signal)
theta_freq, theta_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 1), clean_signal)
phi_freq, phi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 2), clean_signal)
euler_history = result2array(euler_history)
euler_history[:,1:] = bring_inside_bounds(euler_history[:,1:], 0.0, TWOPI)

submartian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [4, 5]))
submartian_point[:,1:] = bring_inside_bounds(submartian_point[:,1:], -PI, PI, include = 'upper')

# theta = result2array(extract_elements_from_history(simulator.dependent_variable_history, 11))
# theta[:,1] = bring_inside_bounds(theta[:,1], 0.0, TWOPI)

plot_kepler_elements(keplerian_history)

plt.figure()
plt.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label = r'$\psi$')
plt.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label = r'$\theta$')
plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
plt.legend()
plt.grid()
plt.title('Undamped Euler angles')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Angle [º]')

plt.figure()
plt.plot(epochs_array / 86400.0, euler_history[:,1] * 360.0 / TWOPI, label = r'$\psi$')
plt.plot(epochs_array / 86400.0, euler_history[:,2] * 360.0 / TWOPI, label = r'$\theta$')
plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
plt.legend()
plt.grid()
plt.title('Damped Euler angles')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Angle [º]')

plt.figure()
plt.plot(epochs_array / 86400.0, submartian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
plt.plot(epochs_array / 86400.0, submartian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
plt.legend()
plt.grid()
plt.title('Sub-martian point (damped)')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Coordinate [º]')

plt.figure()
plt.plot(epochs_array / 86400.0, submartian_point_undamped[:,1] * 360.0 / TWOPI, label = r'$Lat$')
plt.plot(epochs_array / 86400.0, submartian_point_undamped[:,2] * 360.0 / TWOPI, label = r'$Lon$')
plt.legend()
plt.grid()
plt.title('Sub-martian point (undamped)')
plt.xlabel('Time [days since J2000]')
plt.ylabel('Coordinate [º]')

# coeffs = polyfit(epochs_array, submartian_point_undamped[:,2], 1)
# print('')
# print('Mean rotation: ' + str(phobos_mean_rotational_rate) + ' rad/s')
# print('Longitude offset: ' + str(coeffs[0]) + ' rad/s')
# print('Longitude drift: ' + str(coeffs[1]) + ' rad/s')

plt.figure()
plt.scatter(submartian_point[:,2] * 360.0 / TWOPI, submartian_point[:,1] * 360.0 / TWOPI, label= 'Damped')
plt.scatter(submartian_point_undamped[:,2] * 360.0 / TWOPI, submartian_point_undamped[:,1] * 360.0 / TWOPI, label = 'Undamped')
plt.legend()
plt.grid()
# plt.ylim([-8.5, 11.0])
# plt.xlim([-5.0, 5.0])
plt.title(r'Sub-martian point')
# plt.title(r'Sub-martian point (reference)')
plt.xlabel('LON [º]')
plt.ylabel('LAT [º]')

plt.figure()
plt.plot(psi_freq_undamped * 86400.0, psi_amp_undamped * 360 / TWOPI, label = r'$\psi$')
plt.plot(theta_freq_undamped * 86400.0, theta_amp_undamped * 360 / TWOPI, label = r'$\theta$')
plt.plot(phi_freq_undamped * 86400.0, phi_amp_undamped * 360 / TWOPI, label = r'$\phi$')
plt.title(r'Undamped frequency content')
plt.xlabel(r'$\omega$ [rad/day]')
plt.ylabel(r'$A [º]$')
plt.grid()
plt.legend()

plt.figure()
plt.plot(psi_freq * 86400.0, psi_amp * 360 / TWOPI, label = r'$\psi$')
plt.plot(theta_freq * 86400.0, theta_amp * 360 / TWOPI, label = r'$\theta$')
plt.plot(phi_freq * 86400.0, phi_amp * 360 / TWOPI, label = r'$\phi$')
plt.title(r'Damped frequency content')
plt.xlabel(r'$\omega$ [rad/day]')
plt.ylabel(r'$A [º]$')
plt.grid()
plt.legend()

# plt.figure()
# plt.plot(epochs_array / 86400.0, theta[:,1] * 360.0 / TWOPI)
# plt.grid()
# plt.title('True anomaly')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$\theta$ [º]')