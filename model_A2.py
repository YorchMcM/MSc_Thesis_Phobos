'''
In this script we will define model A2. It includes:

· Translational model: the states output by model A1
· Initial epoch: 01/01/2000 at 15:37:15 (first periapsis passage)
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
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

# LOAD SPICE KERNELS
spice.load_standard_kernels()

print('Creating Solar System...')
# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
trajectory_file = './everything-works-results/model-a1/perturbed-baseline.txt'
imposed_trajectory = read_vector_history_from_file(trajectory_file)
phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source)

print('Defining propagation...')
# DEFINE PROPAGATION
initial_epoch = 13035.0  # This is (approximately) the first periapsis passage since J2000
simulation_time = 90.0 * constants.JULIAN_DAY
dependent_variables_to_save = [ propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos'),  # 0, 1, 2
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),  # 6, 7, 8, 9, 10, 11
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                propagation_setup.dependent_variable.relative_position('Phobos', 'Mars'),  # 15, 16, 17
                                torque_norm_from_body_on_phobos('Sun'),  # 18
                                torque_norm_from_body_on_phobos('Earth'),  # 19
                                # torque_norm_from_body_on_phobos('Moon'),  # 20
                                torque_norm_from_body_on_phobos('Mars'),  # 21
                                torque_norm_from_body_on_phobos('Deimos'),  # 22
                                torque_norm_from_body_on_phobos('Jupiter')  # 23
                                # torque_norm_from_body_on_phobos('Saturn')  # 24
                                ]

print('Simulating undamped dynamics...')
# SIMULATE UNDAMPED DYNAMICS FOR REFERENCE
phobos_mean_rotational_rate = 0.000228035245  # In rad/s (more of this number, longitude slope goes down)
fake_initial_state = get_fake_initial_state(bodies, initial_epoch, phobos_mean_rotational_rate)
fake_propagator_settings = get_model_a2_propagator_settings(bodies, simulation_time, fake_initial_state, dependent_variables_to_save)
simulator_undamped = numerical_simulation.create_dynamics_simulator(bodies, fake_propagator_settings)

print('Obtaining damped initial state...')
# OBTAIN DAMPED INITIAL STATE
#                                  4h, 8h, 16h, 1d 8h, 2d 16h, 5d 8h, 10d 16h, 21d 8h, 42d 16h, 85d 8h  // Up to 853d 8h in get_zero_proper_mode function
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0])*3600.0)  # In seconds.
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         fake_propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)
damped_initial_state = damping_results.initial_state

print('Simulating damped dynamics...')
# SIMULATE DAMPED DYNAMICS
propagator_settings = get_model_a2_propagator_settings(bodies, simulation_time, damped_initial_state, dependent_variables_to_save)
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

print('ALL SIMULATIONS FINISHED')

# POST PROCESS (CHECKS)
checks = [0, 0, 0, 0, 0, 1]
mars_mu = bodies.get('Mars').gravitational_parameter
epochs_array = np.array(list(simulator.state_history.keys()))

# Trajectory
if checks[0]:
    cartesian_history_undamped = extract_elements_from_history(simulator_undamped.dependent_variable_history, [15, 16, 17])
    trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
    cartesian_history = extract_elements_from_history(simulator.dependent_variable_history, [15, 16, 17])
    trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

# Orbit does not blow up.
if checks[1]:
    keplerian_history_undamped = extract_elements_from_history(simulator_undamped.dependent_variable_history, [6, 7, 8, 9, 10, 11])
    plot_kepler_elements(keplerian_history_undamped, title = 'Undamped COEs')
    keplerian_history = extract_elements_from_history(simulator.dependent_variable_history, [6, 7, 8, 9, 10, 11])
    plot_kepler_elements(keplerian_history, title = 'Damped COEs')

# Orbit is equatorial
if checks[2]:
    sub_phobian_point_undamped = result2array(extract_elements_from_history(simulator_undamped.dependent_variable_history, [13, 14]))
    sub_phobian_point_undamped[:,1:] = bring_inside_bounds(sub_phobian_point_undamped[:,1:], -PI, PI, include = 'upper')
    sub_phobian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [13, 14]))
    sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:,1:], -PI, PI, include = 'upper')

    plt.figure()
    plt.scatter(sub_phobian_point_undamped[:,2] * 360.0 / TWOPI, sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label = 'Undamped')
    plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:,1] * 360.0 / TWOPI, label = 'Damped', marker = '+')
    plt.grid()
    plt.title('Sub-phobian point')
    plt.xlabel('LON [º]')
    plt.ylabel('LAT [º]')
    plt.legend()

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.title('Undamped sub-phobian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.title('Damped sub-phobian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

# Phobos' Euler angles. In a torque-free environment, the first two are constant and the third grows linearly as
# indicated by the angular speed. This happens both in the undamped and damped cases. In an environment with torques,
# the undamped angles contain free modes while the damped ones do not.
if checks[3]:
    normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor, phobos_mean_rotational_rate)
    clean_signal = [TWOPI, 1]
    euler_history_undamped = result2array(extract_elements_from_history(simulator_undamped.dependent_variable_history, [2, 1, 0]))
    euler_history_undamped[:,1:] = -euler_history_undamped[:,1:]
    euler_history_undamped = bring_history_inside_bounds(array2result(euler_history_undamped), 0.0, TWOPI)
    psi_freq_undamped, psi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 0), clean_signal)
    theta_freq_undamped, theta_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 1), clean_signal)
    phi_freq_undamped, phi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 2), clean_signal)
    euler_history = result2array(extract_elements_from_history(simulator.dependent_variable_history, [2, 1, 0]))
    euler_history[:,1:] = -euler_history[:,1:]
    euler_history = bring_history_inside_bounds(array2result(euler_history), 0.0, TWOPI)
    psi_freq, psi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 0), clean_signal)
    theta_freq, theta_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 1), clean_signal)
    phi_freq, phi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 2), clean_signal)
    euler_history_undamped = result2array(euler_history_undamped)
    euler_history = result2array(euler_history)

    plt.figure()
    plt.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label = r'$\psi$')
    plt.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label = r'$\theta$')
    plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
    plt.legend()
    plt.grid()
    plt.xlim([0.0, 3.5])
    plt.title('Undamped Euler angles')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Angle [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, euler_history[:,1] * 360.0 / TWOPI, label = r'$\psi$')
    plt.plot(epochs_array / 86400.0, euler_history[:,2] * 360.0 / TWOPI, label = r'$\theta$')
    plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
    plt.legend()
    plt.grid()
    plt.xlim([0.0, 3.5])
    plt.title('Damped Euler angles')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Angle [º]')

    plt.figure()
    plt.plot(psi_freq_undamped * 86400.0, psi_amp_undamped * 360 / TWOPI, label = r'$\psi$')
    plt.plot(theta_freq_undamped * 86400.0, theta_amp_undamped * 360 / TWOPI, label = r'$\theta$')
    plt.plot(phi_freq_undamped * 86400.0, phi_amp_undamped * 360 / TWOPI, label = r'$\phi$')
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobian mean motion')
    plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
    plt.title(r'Undamped frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.xlim([0, 50])
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(psi_freq * 86400.0, psi_amp * 360 / TWOPI, label = r'$\psi$')
    plt.plot(theta_freq * 86400.0, theta_amp * 360 / TWOPI, label = r'$\theta$')
    plt.plot(phi_freq * 86400.0, phi_amp * 360 / TWOPI, label = r'$\phi$')
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobian mean motion')
    plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
    plt.title(r'Damped frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.xlim([0, 50])
    plt.grid()
    plt.legend()

# Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
if checks[4]:
    sub_martian_point_undamped = result2array(extract_elements_from_history(simulator_undamped.dependent_variable_history, [4, 5]))
    sub_martian_point_undamped[:,1:] = bring_inside_bounds(sub_martian_point_undamped[:,1:], -PI, PI, include = 'upper')
    libration_history_undamped = extract_elements_from_history(simulator_undamped.dependent_variable_history, 5)
    libration_freq_undamped, libration_amp_undamped = get_fourier_elements_from_history(libration_history_undamped)
    sub_martian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [4, 5]))
    sub_martian_point[:,1:] = bring_inside_bounds(sub_martian_point[:,1:], -PI, PI, include = 'upper')
    libration_history = extract_elements_from_history(simulator.dependent_variable_history, 5)
    libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)

    plt.figure()
    plt.scatter(sub_martian_point_undamped[:,2] * 360.0 / TWOPI, sub_martian_point_undamped[:,1] * 360.0 / TWOPI, label = 'Undamped')
    plt.scatter(sub_martian_point[:,2] * 360.0 / TWOPI, sub_martian_point[:,1] * 360.0 / TWOPI, label = 'Damped', marker = '+')
    plt.grid()
    plt.title(r'Sub-martian point')
    plt.xlabel('LON [º]')
    plt.ylabel('LAT [º]')
    plt.legend()

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.xlim([0.0, 3.5])
    # plt.title('Undamped sub-martian point ($\omega = ' + str(phobos_mean_rotational_rate) + '$ rad/s)')
    plt.title('Undamped sub-martian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_martian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_martian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.xlim([0.0, 3.5])
    plt.title('Damped sub-martian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

    plt.figure()
    plt.semilogy(libration_freq_undamped * 86400.0, libration_amp_undamped * 360 / TWOPI)
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobian mean motion')
    plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
    plt.title(r'Undamped libration frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.grid()
    plt.xlim([0, 21])
    plt.legend()

    plt.figure()
    plt.semilogy(libration_freq * 86400.0, libration_amp * 360 / TWOPI)
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobian mean motion')
    plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
    plt.title(r'Damped libration frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.grid()
    plt.xlim([0, 21])
    plt.legend()

if checks[5]:

    # third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
    third_bodies = ['Sun', 'Earth', 'Mars', 'Deimos', 'Jupiter']

    third_body_torques_undamped = result2array(extract_elements_from_history(simulator_undamped.dependent_variable_history, list(range(18,23))))
    plt.figure()
    for idx, body in enumerate(third_bodies):
        plt.semilogy(epochs_array / 86400.0, third_body_torques_undamped[:,idx+1], label = body)
    plt.title('Third body torques (undamped rotation)')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel(r'Torque [N$\cdot$m]')
    plt.legend()
    plt.grid()

    third_body_torques = result2array(extract_elements_from_history(simulator.dependent_variable_history, list(range(18,23))))
    plt.figure()
    for idx, body in enumerate(third_bodies):
        plt.semilogy(epochs_array / 86400.0, third_body_torques[:,idx+1], label = body)
    plt.title('Third body torques')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel(r'Torque [N$\cdot$m]')
    plt.legend()
    plt.grid()