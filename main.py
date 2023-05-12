from Auxiliaries import *
import myconstants

average_mean_motion = myconstants.average_mean_motion
normal_mode = myconstants.normal_mode
phobos_mean_rotational_rate = myconstants.phobos_mean_rotational_rate

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
trajectory_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
imposed_trajectory = extract_elements_from_history(read_vector_history_from_file(trajectory_file), [0, 1, 2, 3, 4, 5])
phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source, scaled_amplitude)

# dissipation_times = [4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0, 8192.0]
# first_dissipation_index = 4
# last_dissipation_index = None
#
# if first_dissipation_index is None and last_dissipation_index is None:
#     dissipation_slash = dissipation_times
#     array_size = len(dissipation_times)
# elif first_dissipation_index is None:
#     dissipation_slash = dissipation_times[:last_dissipation_index]
#     array_size = len(dissipation_times) - last_dissipation_index
# elif last_dissipation_index is None:
#     dissipation_slash = dissipation_times[first_dissipation_index:]
#     array_size = len(dissipation_times) - first_dissipation_index
# else:
#     dissipation_slash = dissipation_times[first_dissipation_index:last_dissipation_index]
#     array_size = last_dissipation_index - first_dissipation_index

read_dir = getcwd() + '/estimation-ab/alpha/'
# # freq_undamped, amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(
# #     read_vector_history_from_file(read_dir + 'dependents-undamped.txt'), 5))
#
# freqs, amps = np.zeros([len(freq_undamped), array_size]), np.zeros([len(freq_undamped), array_size])
# for idx, time in enumerate(dissipation_slash):
#     print(idx, time)
#     current_file = read_dir + 'dependents-d' + str(int(time)) + '-full.txt'
#     # if idx < len(dissipation_times)-1: current_file = current_file + '-full'
#     # current_file = current_file + '.txt'
#     freqs[:,idx], amps[:,idx] = get_fourier_elements_from_history(extract_elements_from_history(
#         read_vector_history_from_file(current_file), 5))
#
# fig = plt.figure()
# # plt.loglog(freq_undamped * 86400.0, amp_undamped * 360 / TWOPI, c = 'k', marker='.', label = 'Undamped')
# for idx, time in enumerate(dissipation_slash):
#     if time == 4: plt.loglog(freqs[:,idx] * 86400.0, amps[:,idx] * 360 / TWOPI, linestyle = '--', marker='.', label = str(int(time)) + 'h')
#     else: plt.loglog(freqs[:, idx] * 86400.0, amps[:, idx] * 360 / TWOPI, marker='.', label=str(int(time)) + 'h')
# plt.axline((0, 0.0014), (1, 0.0014), ls='dashed', c='k', label='Rambaux\'s threshold')
# plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r',
#            label='Phobos\' mean motion (and integer multiples)')
# plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='b', label='Longitudinal normal mode')
# plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.title(r'Libration frequency content for different damping times')
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'$A [º]$')
# plt.grid()
# fig.legend()

# sma = result2array(extract_elements_from_history(read_vector_history_from_file(read_dir + 'dependents-undamped.txt'), 6))
# print('Undamped: ', (sma[-1,0] - sma[0,0]) / 3600.0, 'h')
# sma_all = np.zeros([len(sma[:,1]), 12])
# for idx, time in enumerate(dissipation_times):
#     # print(idx, time)
#     current_file = read_dir + 'dependents-d' + str(int(time)) + '-full.txt'
#     # if idx < len(dissipation_times)-1: current_file = current_file + '-full'
#     # current_file = current_file + '.txt'
#     temp = result2array(extract_elements_from_history(read_vector_history_from_file(current_file), 6))
#     print('d' + str(time) + '-full: ', (temp[-1,0] - temp[0,0]) / 3600.0, 'h')
#     sma_all[:, idx] = temp[:, 1]
#
# fig = plt.figure()
# plt.plot(sma[:,0] / 86400.0, sma[:,1] / 1000.0, marker='.', label = 'Undamped')
# for idx, time in enumerate(dissipation_times):
#     plt.plot(sma[:,0] / 86400.0, sma_all[:,idx] / 1000.0, marker='.', label = str(int(time)) + 'h')
# plt.title(r'Semimajor axis for different damping times')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel('$a$ [km]')
# plt.grid()
# fig.legend()

# freq_lib, amp_lib = get_fourier_elements_from_history(extract_elements_from_history(
#     read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt'), 5))
# freq_a, amp_a = get_fourier_elements_from_history(extract_elements_from_history(
#     read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt'), 6))
# freq_e, amp_e = get_fourier_elements_from_history(extract_elements_from_history(
#     read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt'), 7))
# freq_i, amp_i = get_fourier_elements_from_history(extract_elements_from_history(
#     read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt'), 8))
# freq_omega, amp_omega = get_fourier_elements_from_history(extract_elements_from_history(
#     read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt'), 9), clean_signal = [TWOPI, 1])
# freq_raan, amp_raan = get_fourier_elements_from_history(extract_elements_from_history(
#     read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt'), 10))
#
# fig = plt.figure()
# plt.loglog(freq_lib * 86400.0, amp_lib * 360 / TWOPI, marker='.', label = 'Libration [º]')
# # plt.loglog(freq_a * 86400.0, amp_a / 1000.0, marker='.', label = 'Semi-major axis [km]')
# # plt.loglog(freq_e * 86400.0, amp_e, marker='.', label = 'Eccentricity [-]')
# # plt.loglog(freq_i * 86400.0, amp_i * 360 / TWOPI, marker='.', label = 'Inclination [º]')
# # plt.loglog(freq_raan * 86400.0, amp_raan * 360 / TWOPI, marker='.', label = r'$\Omega$ [º]')
# plt.loglog(freq_omega * 86400.0, amp_omega * 360 / TWOPI, marker='.', label = r'$\omega$ [º]')
# plt.axline((0, 0.0014), (1, 0.0014), ls='dashed', c='k', label='Rambaux\'s threshold')
# plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r',
#            label='Phobos\' mean motion (and integer multiples)')
# plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='b', label='Longitudinal normal mode')
# plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.title(r'Frequency content of different magnitudes')
# plt.xlabel(r'Frequency [rad/day]')
# plt.ylabel(r'$A [-]$')
# plt.grid()
# plt.legend()

# see_undamped = False
# checks = [0, 0, 0, 1, 0, 0]
# if sum(checks) > 0:
#     mars_mu = myconstants.mars_mu
#     states_undamped = extract_elements_from_history(read_vector_history_from_file(read_dir + 'states-undamped.txt'), [3, 4, 5, 6, 7])
#     dependents_undamped = read_vector_history_from_file(read_dir + 'dependents-undamped.txt')
#     states_damped = extract_elements_from_history(read_vector_history_from_file(read_dir + 'states-d8192-full.txt'), [3, 4, 5, 6, 7])
#     dependents_damped = read_vector_history_from_file(read_dir + 'dependents-d8192-full.txt')
#     epochs_array = np.array(list(states_damped.keys()))
#
# # Trajectory
# if checks[0]:
#     if see_undamped:
#         cartesian_history_undamped = extract_elements_from_history(read_vector_history_from_file(read_dir + 'states-undamped.txt'), [0, 1, 2])
#         trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
#     cartesian_history = extract_elements_from_history(dependents_damped, [15, 16, 17])
#     trajectory_3d(cartesian_history, ['Phobos'], 'Mars')
#
# # Orbit does not blow up.
# if checks[1]:
#     if see_undamped:
#         keplerian_history_undamped = extract_elements_from_history(dependents_undamped, [6, 7, 8, 9, 10, 11])
#         plot_kepler_elements(keplerian_history_undamped, title = 'Undamped COEs')
#     keplerian_history = extract_elements_from_history(dependents_damped, [6, 7, 8, 9, 10, 11])
#     plot_kepler_elements(keplerian_history, title = 'Damped COEs')
#
# # Orbit is equatorial
# if checks[2]:
#     if see_undamped:
#         sub_phobian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [13, 14]))
#         sub_phobian_point_undamped[:,1:] = bring_inside_bounds(sub_phobian_point_undamped[:,1:], -PI, PI, include = 'upper')
#     sub_phobian_point = result2array(extract_elements_from_history(dependents_damped, [13, 14]))
#     sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:,1:], -PI, PI, include = 'upper')
#
#     if see_undamped:
#         plt.figure()
#         plt.scatter(sub_phobian_point_undamped[:,2] * 360.0 / TWOPI, sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label = 'Undamped')
#         plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:,1] * 360.0 / TWOPI, label = 'Damped', marker = '+')
#         plt.grid()
#         plt.title('Sub-phobian point')
#         plt.xlabel('LON [º]')
#         plt.ylabel('LAT [º]')
#         plt.legend()
#
#         plt.figure()
#         plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label = r'$Lat$')
#         plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,2] * 360.0 / TWOPI, label = r'$Lon$')
#         plt.legend()
#         plt.grid()
#         plt.title('Undamped sub-phobian point')
#         plt.xlabel('Time [days since J2000]')
#         plt.ylabel('Coordinate [º]')
#
#     plt.figure()
#     plt.plot(epochs_array / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
#     plt.plot(epochs_array / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
#     plt.legend()
#     plt.grid()
#     plt.title('Damped sub-phobian point')
#     plt.xlabel('Time [days since J2000]')
#     plt.ylabel('Coordinate [º]')
#
# # Phobos' Euler angles. In a torque-free environment, the first two are constant and the third grows linearly as
# # indicated by the angular speed. This happens both in the undamped and damped cases. In an environment with torques,
# # the undamped angles contain free modes while the damped ones do not.
# if checks[3]:
#     normal_mode = myconstants.normal_mode
#     clean_signal = [TWOPI, 1]
#     if see_undamped:
#         euler_history_undamped = result2array(extract_elements_from_history(dependents_undamped, [0, 1, 2]))
#         euler_history_undamped = bring_history_inside_bounds(array2result(euler_history_undamped), 0.0, TWOPI)
#         psi_freq_undamped, psi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 0), clean_signal)
#         theta_freq_undamped, theta_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 1), clean_signal)
#         phi_freq_undamped, phi_amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped, 2), clean_signal)
#         euler_history_undamped = result2array(euler_history_undamped)
#     euler_history = result2array(extract_elements_from_history(dependents_damped, [0, 1, 2]))
#     euler_history = bring_history_inside_bounds(array2result(euler_history), 0.0, TWOPI)
#     psi_freq, psi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 0), clean_signal)
#     theta_freq, theta_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 1), clean_signal)
#     phi_freq, phi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 2), clean_signal)
#     euler_history = result2array(euler_history)
#
#     if see_undamped:
#         fig, ax1 = plt.subplots()
#         ax2 = ax1.twinx()
#         ax1.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label = r'$\psi$')
#         ax2.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label = r'$\theta$', c = '#D95319')
#         ax1.set_ylabel(r'$\psi$ [º]')
#         ax2.set_ylabel(r'$\theta$ [º]')
#         ax1.tick_params(axis='y', colors='#0072BD')
#         ax2.tick_params(axis='y', colors='#D95319')
#         ax1.yaxis.label.set_color('#0072BD')
#         ax2.yaxis.label.set_color('#D95319')
#         ax1.spines['right'].set_color('#0072BD')
#         ax2.spines['right'].set_color('#D95319')
#         ax1.set_xlabel('Time [days since J2000]')
#         ax1.grid()
#         ax1.set_title('Undamped Euler angles')
#
#         # plt.figure()
#         # # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label = r'$\psi$')
#         # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label = r'$\theta$')
#         # # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
#         # plt.legend()
#         # plt.grid()
#         # # plt.xlim([0.0, 3.5])
#         # plt.title('Undamped Euler angles')
#         # plt.xlabel('Time [days since J2000]')
#         # plt.ylabel('Angle [º]')
#
#     fig, ax1 = plt.subplots()
#     ax1.plot(epochs_array / 86400.0, euler_history[:, 1] * 360.0 / TWOPI, label=r'$\psi$')
#     set_axis_color(ax1, 'left', '#0072BD')
#     ax1.set_ylabel(r'$\psi$ [º]')
#     ax2 = ax1.twinx()
#     ax2.plot(epochs_array / 86400.0, euler_history[:, 2] * 360.0 / TWOPI, label=r'$\theta$', c = '#D95319')
#     set_axis_color(ax2, 'right', '#D95319')
#     ax2.set_ylabel(r'$\theta$ [º]')
#     # ax1.tick_params(axis='y', colors = '#0072BD')
#     # ax2.tick_params(axis='y', colors = '#D95319')
#     # ax1.spines['left'].set_color('#0072BD')
#     # ax2.spines['right'].set_color('#D95319')
#     # ax1.yaxis.label.set_color('#0072BD')
#     # ax2.yaxis.label.set_color('#D95319')
#     ax1.set_xlabel('Time [days since J2000]')
#     ax1.grid()
#     ax1.set_title('Damped Euler angles')
#
#     # plt.figure()
#     # # plt.plot(epochs_array / 86400.0, euler_history[:,1] * 360.0 / TWOPI, label = r'$\psi$')
#     # plt.plot(epochs_array / 86400.0, euler_history[:,2] * 360.0 / TWOPI, label = r'$\theta$')
#     # # plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
#     # plt.legend()
#     # plt.grid()
#     # # plt.xlim([0.0, 3.5])
#     # plt.title('Damped Euler angles')
#     # plt.xlabel('Time [days since J2000]')
#     # plt.ylabel('Angle [º]')
#
#     if see_undamped:
#         plt.figure()
#         plt.loglog(psi_freq_undamped * 86400.0, psi_amp_undamped * 360 / TWOPI, label = r'$\psi$', marker = '.')
#         plt.loglog(theta_freq_undamped * 86400.0, theta_amp_undamped * 360 / TWOPI, label = r'$\theta$', marker = '.')
#         plt.loglog(phi_freq_undamped * 86400.0, phi_amp_undamped * 360 / TWOPI, label = r'$\phi$', marker = '.')
#         plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
#         plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
#         plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#         plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#         plt.title(r'Undamped frequency content')
#         plt.xlabel(r'$\omega$ [rad/day]')
#         plt.ylabel(r'$A [º]$')
#         # plt.xlim([0, 70])
#         plt.grid()
#         plt.legend()
#
#     plt.figure()
#     plt.loglog(psi_freq * 86400.0, psi_amp * 360 / TWOPI, label = r'$\psi$', marker = '.')
#     plt.loglog(theta_freq * 86400.0, theta_amp * 360 / TWOPI, label = r'$\theta$', marker = '.')
#     plt.loglog(phi_freq * 86400.0, phi_amp * 360 / TWOPI, label = r'$\phi$', marker = '.')
#     plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
#     plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
#     plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#     plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#     plt.title(r'Damped frequency content')
#     plt.xlabel(r'$\omega$ [rad/day]')
#     plt.ylabel(r'$A [º]$')
#     # plt.xlim([0, 70])
#     plt.grid()
#     plt.legend()
#
# # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
# if checks[4]:
#     if see_undamped:
#         sub_martian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [4, 5]))
#         sub_martian_point_undamped[:,1:] = bring_inside_bounds(sub_martian_point_undamped[:,1:], -PI, PI, include = 'upper')
#         libration_history_undamped = extract_elements_from_history(dependents_undamped, 5)
#         libration_freq_undamped, libration_amp_undamped = get_fourier_elements_from_history(libration_history_undamped)
#     sub_martian_point = result2array(extract_elements_from_history(dependents_damped, [4, 5]))
#     sub_martian_point[:,1:] = bring_inside_bounds(sub_martian_point[:,1:], -PI, PI, include = 'upper')
#     libration_history = extract_elements_from_history(dependents_damped, 5)
#     libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)
#
#     if see_undamped:
#         plt.figure()
#         plt.scatter(sub_martian_point_undamped[:,2] * 360.0 / TWOPI, sub_martian_point_undamped[:,1] * 360.0 / TWOPI, label = 'Undamped')
#         plt.scatter(sub_martian_point[:,2] * 360.0 / TWOPI, sub_martian_point[:,1] * 360.0 / TWOPI, label = 'Damped', marker = '+')
#         plt.grid()
#         plt.title(r'Sub-martian point')
#         plt.xlabel('LON [º]')
#         plt.ylabel('LAT [º]')
#         plt.legend()
#
#         plt.figure()
#         plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:,1] * 360.0 / TWOPI, label = r'$Lat$')
#         plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:,2] * 360.0 / TWOPI, label = r'$Lon$')
#         plt.legend()
#         plt.grid()
#         # plt.title('Undamped sub-martian point ($\omega = ' + str(phobos_mean_rotational_rate) + '$ rad/s)')
#         plt.title('Undamped sub-martian point')
#         plt.xlabel('Time [days since J2000]')
#         plt.ylabel('Coordinate [º]')
#
#     plt.figure()
#     plt.plot(epochs_array / 86400.0, sub_martian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
#     plt.plot(epochs_array / 86400.0, sub_martian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
#     plt.legend()
#     plt.grid()
#     plt.title('Damped sub-martian point')
#     plt.xlabel('Time [days since J2000]')
#     plt.ylabel('Coordinate [º]')
#
#     if see_undamped:
#         plt.figure()
#         plt.loglog(libration_freq_undamped * 86400.0, libration_amp_undamped * 360 / TWOPI, marker = '.')
#         plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
#         plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
#         plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#         plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#         plt.title(r'Undamped libration frequency content')
#         plt.xlabel(r'$\omega$ [rad/day]')
#         plt.ylabel(r'$A [º]$')
#         plt.grid()
#         plt.legend()
#
#     plt.figure()
#     plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker = '.')
#     plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion (and integer multiples)')
#     plt.axline((normal_mode * 86400.0, 0),(normal_mode * 86400.0, 1), ls = 'dashed', c = 'k', label = 'Longitudinal normal mode')
#     plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#     plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
#     plt.title(r'Damped libration frequency content')
#     plt.xlabel(r'$\omega$ [rad/day]')
#     plt.ylabel(r'$A [º]$')
#     plt.grid()
#     plt.legend()
#
# # Torques exerted by third bodies
# if checks[5]:
#
#     # third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
#     third_bodies = ['Sun', 'Earth', 'Mars', 'Deimos', 'Jupiter']
#
#     if see_undamped:
#         third_body_torques_undamped = result2array(extract_elements_from_history(dependents_undamped, list(range(18,23))))
#         plt.figure()
#         for idx, body in enumerate(third_bodies):
#             plt.semilogy(epochs_array / 86400.0, third_body_torques_undamped[:,idx+1], label = body)
#         plt.title('Third body torques (undamped rotation)')
#         plt.xlabel('Time [days since J2000]')
#         plt.ylabel(r'Torque [N$\cdot$m]')
#         plt.yticks([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
#         plt.legend()
#         plt.grid()
#
#     third_body_torques = result2array(extract_elements_from_history(dependents_damped, list(range(18,23))))
#     plt.figure()
#     for idx, body in enumerate(third_bodies):
#         plt.semilogy(epochs_array / 86400.0, third_body_torques[:,idx+1], label = body)
#     plt.title('Third body torques')
#     plt.xlabel('Time [days since J2000]')
#     plt.ylabel(r'Torque [N$\cdot$m]')
#     plt.yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
#     plt.legend()
#     plt.grid()

# mars_mu = myconstants.mars_mu
# dependents_undamped_a2 = read_vector_history_from_file(read_dir + 'model-a2/dependents-undamped.txt')
# dependents_damped_a2 = read_vector_history_from_file(read_dir + 'model-a2/dependents-d8192-full.txt')
# dependents_undamped_b = read_vector_history_from_file(read_dir + 'model-b/dependents-undamped.txt')
# dependents_damped_b = read_vector_history_from_file(read_dir + 'model-b/dependents-d8192-full.txt')
# epochs_array = np.array(list(dependents_damped_b.keys()))
# normal_mode = myconstants.normal_mode
# clean_signal = [TWOPI, 1]

# MODEL A2
# euler_history_undamped_a2 = bring_history_inside_bounds(extract_elements_from_history(dependents_undamped_a2, [0, 1, 2]), 0.0, TWOPI)
# psi_freq_undamped_a2, psi_amp_undamped_a2 = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped_a2, 0), clean_signal)
# theta_freq_undamped_a2, theta_amp_undamped_a2 = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped_a2, 1), clean_signal)
# phi_freq_undamped_a2, phi_amp_undamped_a2 = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped_a2, 2), clean_signal)
# euler_history_undamped_a2 = result2array(euler_history_undamped_a2)
#
# euler_history_a2 = bring_history_inside_bounds(extract_elements_from_history(dependents_damped_a2, [0, 1, 2]), 0.0, TWOPI)
# psi_freq_a2, psi_amp_a2 = get_fourier_elements_from_history(extract_elements_from_history(euler_history_a2, 0), clean_signal)
# theta_freq_a2, theta_amp_a2 = get_fourier_elements_from_history(extract_elements_from_history(euler_history_a2, 1), clean_signal)
# phi_freq_a2, phi_amp_a2 = get_fourier_elements_from_history(extract_elements_from_history(euler_history_a2, 2), clean_signal)
# euler_history_a2 = result2array(euler_history_a2)

# # MODEL B
# euler_history_undamped_b = bring_history_inside_bounds(extract_elements_from_history(dependents_undamped_b, [0, 1, 2]), 0.0, TWOPI)
# psi_freq_undamped_b, psi_amp_undamped_b = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped_b, 0), clean_signal)
# theta_freq_undamped_b, theta_amp_undamped_b = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped_b, 1), clean_signal)
# phi_freq_undamped_b, phi_amp_undamped_b = get_fourier_elements_from_history(extract_elements_from_history(euler_history_undamped_b, 2), clean_signal)
# euler_history_undamped_b = result2array(euler_history_undamped_b)
#
# euler_history_b = bring_history_inside_bounds(extract_elements_from_history(dependents_damped_b, [0, 1, 2]), 0.0, TWOPI)
# psi_freq_b, psi_amp_b = get_fourier_elements_from_history(extract_elements_from_history(euler_history_b, 0), clean_signal)
# theta_freq_b, theta_amp_b = get_fourier_elements_from_history(extract_elements_from_history(euler_history_b, 1), clean_signal)
# phi_freq_b, phi_amp_b = get_fourier_elements_from_history(extract_elements_from_history(euler_history_b, 2), clean_signal)
# euler_history_b = result2array(euler_history_b)
#
# # UNDAMPED EULER ANGLES (ONLY PSI AND THETA)
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(epochs_array / 86400.0, euler_history_undamped_b[:, 1] * 360.0 / TWOPI, label=r'$\psi$', c='#0072BD')
# ax2.plot(epochs_array / 86400.0, euler_history_undamped_b[:, 2] * 360.0 / TWOPI, label=r'$\theta$', c='#D95319')
# ax1.set_ylabel(r'$\psi$ [º]')
# ax2.set_ylabel(r'$\theta$ [º]')
# ax1.tick_params(axis='y', colors='#0072BD')
# ax2.tick_params(axis='y', colors='#D95319')
# ax1.yaxis.label.set_color('#0072BD')
# ax2.yaxis.label.set_color('#D95319')
# ax2.spines['left'].set_color('#0072BD')
# ax2.spines['right'].set_color('#D95319')
# ax1.set_xlabel('Time [days since J2000]')
# ax1.grid()
# ax1.set_title('Undamped Euler angles')
#
# # DAMPED EULER ANGLES (ONLY PSI AND THETA)
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(epochs_array / 86400.0, euler_history_b[:, 1] * 360.0 / TWOPI, label=r'$\psi$', c='#0072BD')
# ax2.plot(epochs_array / 86400.0, euler_history_b[:, 2] * 360.0 / TWOPI, label=r'$\theta$', c='#D95319')
# ax1.set_ylabel(r'$\psi$ [º]')
# ax2.set_ylabel(r'$\theta$ [º]')
# ax1.tick_params(axis='y', colors='#0072BD')
# ax2.tick_params(axis='y', colors='#D95319')
# ax1.yaxis.label.set_color('#0072BD')
# ax2.yaxis.label.set_color('#D95319')
# ax2.spines['left'].set_color('#0072BD')
# ax2.spines['right'].set_color('#D95319')
# ax1.set_xlabel('Time [days since J2000]')
# ax1.grid()
# ax1.set_title('Damped Euler angles')
#
# # FOURIER TRANSFORM OF ALL THREE UNDAPMPED ANGLES
# plt.figure()
# plt.loglog(psi_freq_undamped_b * 86400.0, psi_amp_undamped_b * 360 / TWOPI, label=r'$\psi$', marker='.')
# plt.loglog(theta_freq_undamped_b * 86400.0, theta_amp_undamped_b * 360 / TWOPI, label=r'$\theta$', marker='.')
# plt.loglog(phi_freq_undamped_b * 86400.0, phi_amp_undamped_b * 360 / TWOPI, label=r'$\phi$', marker='.', markersize = 3.0)
# plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k', label='Longitudinal normal mode')
# plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.title(r'Undamped frequency content')
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'$A [º]$')
# plt.grid()
# plt.legend()
#
# # FOURIER TRANSFORM OF ALL THREE DAMPED ANGLES
# plt.figure()
# plt.loglog(psi_freq_b * 86400.0, psi_amp_b * 360 / TWOPI, label=r'$\psi$', marker='.')
# plt.loglog(theta_freq_b * 86400.0, theta_amp_b * 360 / TWOPI, label=r'$\theta$', marker='.')
# plt.loglog(phi_freq_b * 86400.0, phi_amp_b * 360 / TWOPI, label=r'$\phi$', marker='.', markersize = 3.0)
# plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k', label='Longitudinal normal mode')
# plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
# plt.title(r'Damped frequency content')
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'$A [º]$')
# plt.grid()
# plt.legend()
