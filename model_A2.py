'''
In this script we will define model A2. It includes:

· Translational model: the states output by model A1
· Initial epoch: 01/01/2000 at 15:37:15 (first periapsis passage)
· Initial state: damped initial state provided by Tudat.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
· Torques: Mars' center of mass on Phobos' quadrupole gravity field.
· Propagator: Quaternion and angular velocity vector.

The propagation in model A1 gives an average mean motion of 2.278563609852602e-4 rad/s = 19.68678958912648 rad/day. The
associated orbital period is of 7h 39min 35.20s.
The tweaked rotational motion in model A2 is of 2.28035245e-4 rad/s = 19.702245168 rad/day. The associated rotational
period is of 7h 39min 13.57s.

'''
import os

# IMPORTS
import numpy as np
from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *
from time import time

from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants
from tudatpy.io import save2txt

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
# plt.rc('text', usetex = True)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')

verbose = True
save_results = True
see_undamped = False
average_mean_motion = 0.0002278563609852602
phobos_mean_rotational_rate = 0.000228035245  # In rad/s (more of this number, longitude slope goes down)

#                                  4h,  8h,  16h,  1d 8h, 2d 16h, 5d 8h, 10d 16h, 21d 8h, 42d 16h, 85d 8h, 170d 16h, 341d 8h, 682d 16h  // Up to 6826d 16h in get_zero_proper_mode function
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0,   8192.0])*3600.0)  # In seconds.
# dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0,   8192.0,  16384.0])*3600.0)  # In seconds.

if save_results: save_dir = os.getcwd() + '/everything-works-results/model-a2/'

# LOAD SPICE KERNELS
if verbose: print('Loading kernels...')
spice.load_standard_kernels()

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
if verbose: print('Creating universe...')
trajectory_file = 'phobos-ephemerides-3500.txt'
imposed_trajectory = read_vector_history_from_file(trajectory_file)
phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source)

# DEFINE PROPAGATION
if verbose: print('Setting up propagation...')
initial_epoch = 13035.0  # This is (approximately) the first periapsis passage since J2000
simulation_time = 10.0 * dissipation_times[-1]
euler_angles_wrt_mars_equator_dependent_variable = propagation_setup.dependent_variable.custom_dependent_variable(
    MarsEquatorOfDate(bodies).get_euler_angles_wrt_mars_equator, 3)
dependent_variables_to_save = [ euler_angles_wrt_mars_equator_dependent_variable,  # 0, 1, 2
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),  # 6, 7, 8, 9, 10, 11
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                propagation_setup.dependent_variable.relative_position('Phobos', 'Mars'),  # 15, 16, 17
                                # torque_norm_from_body_on_phobos('Sun'),  # 18
                                # torque_norm_from_body_on_phobos('Earth'),  # 19
                                # torque_norm_from_body_on_phobos('Moon'),  # 20
                                # torque_norm_from_body_on_phobos('Mars'),  # 21
                                # torque_norm_from_body_on_phobos('Deimos'),  # 22
                                # torque_norm_from_body_on_phobos('Jupiter')  # 23
                                # torque_norm_from_body_on_phobos('Saturn')  # 24
                                ]

# OBTAIN DAMPED INITIAL STATE
if verbose: print('Simulating dynamics...')
fake_initial_state = get_fake_initial_state(bodies, initial_epoch, phobos_mean_rotational_rate)
fake_propagator_settings = get_model_a2_propagator_settings(bodies, simulation_time, initial_epoch, fake_initial_state,
                                                            dependent_variables_to_save)
tic = time()
print('Going into the depths of Tudat...')
damping_results = numerical_simulation.propagation.get_zero_proper_mode_rotational_state(bodies,
                                                                                         fake_propagator_settings,
                                                                                         phobos_mean_rotational_rate,
                                                                                         dissipation_times)
tac = time()

if verbose: print('SIMULATIONS FINISHED. Time taken:', (tac-tic) / 60.0, 'minutes.')

if save_results:
    print('Saving damping results...')
    save2txt(damping_results.forward_backward_states[0][0], save_dir + 'states-undamped.txt')
    save2txt(damping_results.forward_backward_dependent_variables[0][0], save_dir + 'dependents-undamped.txt')
    for idx, current_damping_time in enumerate(dissipation_times):
        time_str = str(int(current_damping_time / 3600.0))
        save2txt(damping_results.forward_backward_states[idx+1][1], save_dir + 'states-d' + time_str + '.txt')
        save2txt(damping_results.forward_backward_dependent_variables[idx+1][1], save_dir + 'dependents-d' + time_str + '.txt')
    print('ALL RESULTS SAVED.')

if verbose:
    if save_results: print('Simulating and saving full dynamics for all damping times...')
    else:
        print('Simulating full dynamics for all damping times...')
        print('WARNING: Results for each dissipation time overwrites the results for the preceding one. To keep them all, please save the results.')
tic = time()
for idx, current_damping_time in enumerate(dissipation_times):
    time_str = str(int(current_damping_time / 3600.0))
    print('Simulation ' + str(idx+1) + '/' + str(len(dissipation_times)))
    current_initial_state = damping_results.forward_backward_states[idx][1][initial_epoch]
    current_propagator_settings = get_model_a2_propagator_settings(bodies, simulation_time, current_initial_state, dependent_variables_to_save)
    current_simulator = numerical_simulation.create_dynamics_simulator(bodies, current_propagator_settings)
    save2txt(current_simulator.state_history, save_dir + 'states-d' + time_str + '-full.txt')
    save2txt(current_simulator.dependent_variable_history, save_dir + 'dependents-d' + time_str + '-full.txt')
tac = time()
print('FULL RESULTS FINISHED. Time taken:', (tac-tic) / 60.0, 'minutes.')

# POST PROCESS (CHECKS)
checks = [0, 0, 0, 1, 0, 0]
if sum(checks) > 0:
    mars_mu = bodies.get('Mars').gravitational_parameter
    states_undamped = damping_results.forward_backward_states[0][0]
    dependents_undamped = damping_results.forward_backward_dependent_variables[0][0]
    states_damped = damping_results.forward_backward_states[-1][1]
    dependents_damped = damping_results.forward_backward_dependent_variables[-1][1]
    epochs_array = np.array(list(states_damped.keys()))

# Trajectory
if checks[0]:
    if see_undamped:
        cartesian_history_undamped = extract_elements_from_history(dependents_undamped, [15, 16, 17])
        trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
    cartesian_history = extract_elements_from_history(dependents_damped, [15, 16, 17])
    trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

# Orbit does not blow up.
if checks[1]:
    if see_undamped:
        keplerian_history_undamped = extract_elements_from_history(dependents_undamped, [6, 7, 8, 9, 10, 11])
        plot_kepler_elements(keplerian_history_undamped, title = 'Undamped COEs')
    keplerian_history = extract_elements_from_history(dependents_damped, [6, 7, 8, 9, 10, 11])
    plot_kepler_elements(keplerian_history, title = 'Damped COEs')

# Orbit is equatorial
if checks[2]:
    if see_undamped:
        sub_phobian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [13, 14]))
        sub_phobian_point_undamped[:,1:] = bring_inside_bounds(sub_phobian_point_undamped[:,1:], -PI, PI, include = 'upper')
    sub_phobian_point = result2array(extract_elements_from_history(dependents_damped, [13, 14]))
    sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:,1:], -PI, PI, include = 'upper')

    if see_undamped:
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
    if see_undamped:
        euler_history_undamped = extract_elements_from_history(dependents_undamped, [0, 1, 2])
        euler_history_undamped = bring_history_inside_bounds(euler_history_undamped, 0.0, TWOPI)
        psi_freq_undamped, psi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 0), clean_signal)
        theta_freq_undamped, theta_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 1), clean_signal)
        phi_freq_undamped, phi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 2), clean_signal)
        euler_history_undamped = result2array(euler_history_undamped)
    euler_history = extract_elements_from_history(dependents_damped, [0, 1, 2])
    euler_history = bring_history_inside_bounds(euler_history, 0.0, TWOPI)
    psi_freq, psi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 0), clean_signal)
    theta_freq, theta_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 1), clean_signal)
    phi_freq, phi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 2), clean_signal)
    euler_history = result2array(euler_history)

    if see_undamped:
        plt.figure()
        plt.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label = r'$\psi$')
        plt.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label = r'$\theta$')
        # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
        plt.legend()
        plt.grid()
        # plt.xlim([0.0, 3.5])
        plt.title('Undamped Euler angles')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Angle [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, euler_history[:,1] * 360.0 / TWOPI, label = r'$\psi$')
    plt.plot(epochs_array / 86400.0, euler_history[:,2] * 360.0 / TWOPI, label = r'$\theta$')
    # plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
    plt.legend()
    plt.grid()
    # plt.xlim([0.0, 3.5])
    plt.title('Damped Euler angles')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Angle [º]')

    if see_undamped:
        plt.figure()
        plt.loglog(psi_freq_undamped * 86400.0, psi_amp_undamped * 360 / TWOPI, label = r'$\psi$', marker = '.')
        plt.loglog(theta_freq_undamped * 86400.0, theta_amp_undamped * 360 / TWOPI, label = r'$\theta$', marker = '.')
        plt.loglog(phi_freq_undamped * 86400.0, phi_amp_undamped * 360 / TWOPI, label = r'$\phi$', marker = '.')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
        plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
        plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.title(r'Undamped frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        # plt.xlim([0, 70])
        plt.grid()
        plt.legend()

    plt.figure()
    plt.loglog(psi_freq * 86400.0, psi_amp * 360 / TWOPI, label = r'$\psi$', marker = '.')
    plt.loglog(theta_freq * 86400.0, theta_amp * 360 / TWOPI, label = r'$\theta$', marker = '.')
    plt.loglog(phi_freq * 86400.0, phi_amp * 360 / TWOPI, label = r'$\phi$', marker = '.')
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
    plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
    plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
    plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
    plt.title(r'Damped frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    # plt.xlim([0, 70])
    plt.grid()
    plt.legend()

# Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
if checks[4]:
    if see_undamped:
        sub_martian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [4, 5]))
        sub_martian_point_undamped[:,1:] = bring_inside_bounds(sub_martian_point_undamped[:,1:], -PI, PI, include = 'upper')
        libration_history_undamped = extract_elements_from_history(dependents_undamped, 5)
        libration_freq_undamped, libration_amp_undamped = get_fourier_elements_from_history(libration_history_undamped)
    sub_martian_point = result2array(extract_elements_from_history(dependents_damped, [4, 5]))
    sub_martian_point[:,1:] = bring_inside_bounds(sub_martian_point[:,1:], -PI, PI, include = 'upper')
    libration_history = extract_elements_from_history(dependents_damped, 5)
    libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)

    if see_undamped:
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
        # plt.title('Undamped sub-martian point ($\omega = ' + str(phobos_mean_rotational_rate) + '$ rad/s)')
        plt.title('Undamped sub-martian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_martian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_martian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.title('Damped sub-martian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

    if see_undamped:
        plt.figure()
        plt.loglog(libration_freq_undamped * 86400.0, libration_amp_undamped * 360 / TWOPI, marker = '.')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
        plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
        plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.title(r'Undamped libration frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        plt.grid()
        plt.legend()

    plt.figure()
    plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker = '.')
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
    plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
    plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
    plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
    plt.title(r'Damped libration frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.grid()
    plt.legend()

# Torques exerted by third bodies
if checks[5]:

    # third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
    third_bodies = ['Sun', 'Earth', 'Mars', 'Deimos', 'Jupiter']

    if see_undamped:
        third_body_torques_undamped = result2array(extract_elements_from_history(dependents_undamped, list(range(18,23))))
        plt.figure()
        for idx, body in enumerate(third_bodies):
            plt.semilogy(epochs_array / 86400.0, third_body_torques_undamped[:,idx+1], label = body)
        plt.title('Third body torques (undamped rotation)')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel(r'Torque [N$\cdot$m]')
        plt.yticks([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
        plt.legend()
        plt.grid()

    third_body_torques = result2array(extract_elements_from_history(dependents_damped, list(range(18,23))))
    plt.figure()
    for idx, body in enumerate(third_bodies):
        plt.semilogy(epochs_array / 86400.0, third_body_torques[:,idx+1], label = body)
    plt.title('Third body torques')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel(r'Torque [N$\cdot$m]')
    plt.yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
    plt.legend()
    plt.grid()

print('PROGRAM COMPLETED SUCCESFULLY')