'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 7h 40min
    · Velocity of Phobos : 3 km/s
    · Reference radius: 13 km
'''

from copy import copy, deepcopy
from Auxiliaries import *
from tudatpy.kernel.astro.element_conversion import cartesian_to_spherical


def get_fourier(time_history: np.ndarray, clean_signal: list = [0.0, 0]) -> tuple:

    """This function computes the fast fourier transform of a provided time history. It assumes that the quantity of the time history is real, and calls Numpy's rfft function to compute it. This function complements Numpy's rfft in the following ways:

    · It accounts for a time history with an odd number of entries and removes the last entry to make it of even length.
    · It allows to clean the signal. This encompasses two things:
        - Some quantities present jumps because they are by definition bounded inside an interval, but their evolution is secular. This function removes this jumps and works with a continuous signal.
        - Sometimes one is interested in the residuals of the signal when a predefined polynomial is removed from it. This function allows to remove this polynomial and return the fft of the residuals. The coefficients of the polynomial are computed using Numpy's polyfit.
    · Numpy's rfft returns a complex arrays of coefficients, usually not useful. This function returns the amplitude domain, attending to the fact that (a) the norm of the coefficients is to be taken and (b) the actual amplitude of the sinusoid is twice the norm of the complex coefficient.
    · Numpy's rfftfreq returns a frequency array that is in cycles / unit_of_time. This function returns the frequencies in rad / unit_of_time.

    Parameters
    ----------
    time_history: np.ndarray
        A two-dimensional array with two columns: the first column is the time, the second is the quantity whose frequency content is to be computed.
    clean_signal: list[float]
        This determines (a) whether the signal is to be removed of jumps and (b) whether a polynomial is to be removed from the signal. The first entry of clean_signal is the value of the jumps, and the second entry is the degree of the polynomial.

    Returns
    -------
    tuple
        There are two returns: the array of frequencies (in rad / unit_of_time) and the array of amplitudes.

    """

    if type(clean_signal[1]) != int:
        raise TypeError('(get_fourier): Invalid input. The second entry in clean_signal should be of type "int". A type ' + str(type(clean_signal[1])) + 'was provided.')
    if clean_signal[1] < 0:
        raise ValueError('(get_fourier): Invalid input. The second entry in clean_signal cannot be negative. Current values is ' + str(clean_signal[1]) + '.')
    if clean_signal[0] < 0.0:
        raise ValueError('(get_fourier): Invalid input. The first entry in clean_signal cannot be negative. Current values is ' + str(clean_signal[1]) + '.')

    sample_times = time_history[:,0]
    signal = time_history[:,1]

    if len(sample_times) % 2.0 != 0.0:
        sample_times = sample_times[:-1]
        signal = signal[:-1]

    if clean_signal[0] != 0.0:
        signal = remove_jumps(signal, clean_signal[0])
    if clean_signal[1] != 0:
        coeffs = polyfit(sample_times, signal, clean_signal[1])
        for idx, current_coeff in enumerate(coeffs):
            exponent = idx
            signal = signal - current_coeff*sample_times**exponent

    n = len(sample_times)
    dt = sample_times[1] - sample_times[0]
    frequencies = 2.0*PI * rfftfreq(n, dt)
    amplitudes = 2*abs(rfft(signal, norm = 'forward'))

    return frequencies, amplitudes

# save_dir = os.getcwd() + '/initial-guess-analysis/'
color1, color2, color3, color4 = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']

# bodies = get_solar_system('A2', 'ephemeris/translation-c.eph', 'ephemeris/rotation-c.eph')
# R = MarsEquatorOfDate(bodies).j2000_to_mars_rotation
# full_state_rotation = np.concatenate((np.concatenate((R, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), R), 1)), 0)


# trajectory = read_vector_history_from_file('ephemeris/translation-c.eph')
# read_dir = 'simulation-results/model-b/2023-07-13 12:35:19.273692/'
# trajectory = read_vector_history_from_file(read_dir + 'states-d8192.dat')
# trajectory = extract_elements_from_history(trajectory, [0, 1, 2, 3, 4, 5])
# kepler = read_vector_history_from_file(read_dir + 'dependents-d8192.dat')
# kepler = extract_elements_from_history(kepler, [6, 7, 8, 9, 10, 11])
# epochs_list = list(trajectory.keys())
# state_wrt_mars = np.zeros([len(epochs_list), 7])
# inc_ecc = np.zeros([len(epochs_list), 3])
#
# for idx in range(len(epochs_list)):
#     current_epoch = epochs_list[idx]
#     state_wrt_mars[idx,0] = current_epoch
#     inc_ecc[idx,0] = current_epoch
#
    # state_wrt_mars[idx,1:] = full_state_rotation @ trajectory[current_epoch]
    #
    # r = state_wrt_mars[idx,1:4]
    # v = state_wrt_mars[idx,4:]
#     h = np.cross(r,v)
#
#     inc = np.arccos(h[-1] / np.linalg.norm(h))
#     ecc = np.linalg.norm(((np.cross(v,h))/42.82837e12) - (r / np.linalg.norm(r)))
#
#     inc_ecc[idx,1:] = np.array([inc, ecc])


# theta = np.linspace(0.0, TWOPI, 1001)
# circular_trajectory = np.zeros([len(theta), 3])
# R = np.mean(np.linalg.norm(state_wrt_mars[:,1:4], axis = 1))
# for idx in range(len(theta)):
#     angle = theta[idx]
#     circular_trajectory[idx] = np.array([R*np.cos(angle), R*np.sin(angle), 0.0])
#
# R_mars = 3390e3
# figure, axis = plt.subplots()
# axis.add_patch(plt.Circle((0, 0), R_mars / 1e3, color=color2))
# axis.plot(state_wrt_mars[:,1] / 1e3, state_wrt_mars[:,2] / 1e3, label = 'Real orbit', c = color3)
# axis.plot(circular_trajectory[:,0] / 1e3, circular_trajectory[:,1] / 1e3, label = 'Circular orbit', c = 'purple')
# axis.set_xlabel(r'$x$ [km]')
# axis.set_ylabel(r'$y$ [km]')
# plt.grid()
# plt.legend()
# axis.set_title('Orbit\'s top view')
# plt.axis('equal')

# figure, axis = plt.subplots()
# axis.add_patch(plt.Circle((0, 0), R_mars / 1e3, color=color2))
# axis.plot(state_wrt_mars[:,1] / 1e3, state_wrt_mars[:,3] / 1e3, label = 'Real trajectory', c = color3)
# axis.plot(circular_trajectory[:,0] / 1e3, circular_trajectory[:,2] / 1e3, label = 'Equatorial trajectory', c = 'purple')
# axis.set_xlabel(r'$x$ [km]')
# axis.set_ylabel(r'$z$ [km]')
# plt.grid()
# plt.legend()
# axis.set_title('Side view of Phobos\' trajectory')
# plt.axis('equal')

# figure, axis = plt.subplots()
# axis.add_patch(plt.Circle((0, 0), R_mars / 1e3, color=color2))
# axis.plot(state_wrt_mars[:,2] / 1e3, state_wrt_mars[:,3] / 1e3, label = 'Real orbit', c = color3)
# axis.plot(circular_trajectory[:,1] / 1e3, circular_trajectory[:,2] / 1e3, label = 'Equatorial orbit', c = 'purple')
# axis.set_xlabel(r'$y$ [km]')
# axis.set_ylabel(r'$z$ [km]')
# plt.grid()
# plt.legend()
# axis.set_title('Orbit\'s side view')
# plt.axis('equal')
#
# plot_kepler_elements(kepler, 'Model B')

#
# plt.figure()
# plt.plot(np.array(epochs_list) / constants.JULIAN_DAY, inc_ecc[:,1], label = r'$i$ [rad]')
# plt.plot(np.array(epochs_list) / constants.JULIAN_DAY, inc_ecc[:,2], label = r'$e$ [-]')
# plt.xlabel('Time since J2000 [days]')
# plt.title('Inclination and eccentricity')
# plt.grid()
# plt.legend()

# translation = read_vector_history_from_file('ephemeris/translation-c.eph')
# rotation = read_vector_history_from_file('ephemeris/rotation-c.eph')
# epochs_array = np.array(list(rotation.keys()))
# sph = np.zeros([len(epochs_array), 3])
# sph_tudat = np.zeros([len(epochs_array), 3])
# for idx, epoch in enumerate(epochs_array):
#
#     position = -1.0*translation[epoch][:3]
#     R = quat2mat(rotation[epoch][:4])
#     fake_state = np.concatenate((R.T @ position, np.array([0, 0, 0])))
#     sph[idx] = cartesian_to_spherical(fake_state)[:3]
#
#
# plt.figure()
# plt.plot(epochs_array / constants.JULIAN_DAY, np.degrees(sph[:,2]), label = r'Lon')
# plt.xlabel('Time since J2000 [days]')
# plt.title('Longitude')
# plt.grid()
#
# plt.figure()
# plt.scatter(np.degrees(sph[:,2]), np.degrees(sph[:,1]))
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Coordinates')
# plt.grid()
#
# freq, amp = get_fourier(np.concatenate((np.atleast_2d(epochs_array).T, np.atleast_2d(sph[:,2]).T), axis = 1), [TWOPI, 1])
#
# plt.figure()
# plt.loglog(freq * 86400.0, np.degrees(amp), marker = '.')
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.title(r'$A$ [º]')
# plt.grid()
# plt.title('Longitude frequency content')

# average_mean_motion = 0.0002278563609852602
# dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0,   8192.0])*3600.0)  # In seconds.
# read_dir = os.getcwd() + '/simulation-results/model-a2/2023-07-12 00:55:48.655012/'
# iterations = np.array(list(range(len(dissipation_times)+1)))
# libration_amplitudes = np.zeros(len(iterations))

# bodies = get_solar_system('A2', 'ephemeris/translation-c.eph', 'ephemeris/rotation-c.eph')
# normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor, average_mean_motion)
#
# file = 'simulation-results/model-a1/2023-07-13 10:24:42.226993/dependent-variable-history.dat'
# lib_a1 = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_a1, amp_a1 = get_fourier(lib_a1[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-a2/2023-07-13 11:38:45.319336/dependents-undamped.dat'
# lib_a2_undamped = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_a2_undamped, amp_a2_undamped = get_fourier(lib_a2_undamped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-a2/2023-07-13 11:38:45.319336/dependents-d8192.dat'
# lib_a2_damped = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_a2_damped, amp_a2_damped = get_fourier(lib_a2_damped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-b/2023-07-13 12:35:19.273692/dependents-undamped.dat'
# lib_b_undamped = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_b_undamped, amp_b_undamped = get_fourier(lib_b_undamped[:,[0,2]], [TWOPI, 1])
#
file = 'simulation-results/model-b/2023-07-13 12:35:19.273692/dependents-d8192.dat'
# lib_b_damped = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_b_damped, amp_b_damped = get_fourier(lib_b_damped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-c/2023-07-13 22:02:17.013120/dependents-undamped.dat'
# lib_c_undamped = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_c_undamped, amp_c_undamped = get_fourier(lib_c_undamped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-c/2023-07-13 22:02:17.013120/dependents-d8192.dat'
# lib_c_damped = result2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_c_damped, amp_c_damped = get_fourier(lib_c_damped[:,[0,2]], [TWOPI, 1])
#
# # Aquí van a estar los undamped
# plt.figure()
# plt.scatter(np.degrees(lib_c_undamped[:,2]), np.degrees(lib_c_undamped[:,1]), color = color4, marker = '$C$', label = 'Model C')
# plt.scatter(np.degrees(lib_b_undamped[:,2]), np.degrees(lib_b_undamped[:,1]), color = color3, marker = '$B$', label = 'Model B')
# plt.scatter(np.degrees(lib_a2_undamped[:,2]), np.degrees(lib_a2_undamped[:,1]), color = color2, marker = '$A$', label = 'Model A2')
# plt.grid()
# plt.legend()
# plt.xlabel('Longitude [º]')
# plt.ylabel('Latitude [º]')
# plt.title('Tidal libration (undamped dynamics)')
#
# plt.figure()
# plt.loglog(freq_c_undamped * 86400.0, np.degrees(amp_c_undamped), color = color4, marker = 's', label = 'Model C')
# plt.loglog(freq_b_undamped * 86400.0, np.degrees(amp_b_undamped), color = color3, marker = '+', label = 'Model B')
# plt.loglog(freq_a2_undamped * 86400.0, np.degrees(amp_a2_undamped), color = color2, marker = 'x', label = 'Model A2')
# plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
# plt.axvline(average_mean_motion * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axvline(2.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.axvline(3.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel('A [º]')
# plt.title('Frequency content of libration (undamped dynamics)')
#
# # Y aquí los damped con el modelo A1.
# plt.figure()
# plt.scatter(np.degrees(lib_c_damped[:,2]), np.degrees(lib_c_damped[:,1]), color = color4, marker = '$C$', label = 'Model C')
# plt.scatter(np.degrees(lib_b_damped[:,2]), np.degrees(lib_b_damped[:,1]), color = color3, marker = '$B$', label = 'Model B')
# plt.scatter(np.degrees(lib_a2_damped[:,2]), np.degrees(lib_a2_damped[:,1]), color = color2, marker = '$A2$', label = 'Model A2')
# plt.scatter(np.degrees(lib_a1[:,2]), np.degrees(lib_a1[:,1]), color = color1, marker = '$A1$', label = 'Model A1')
# plt.grid()
# plt.legend()
# plt.xlabel('Longitude [º]')
# plt.ylabel('Latitude [º]')
# plt.title('Tidal libration (damped dynamics)')
#
# plt.figure()
# plt.loglog(freq_c_damped * 86400.0, np.degrees(amp_c_damped), color = color4, marker = 's', label = 'Model C')
# plt.loglog(freq_b_damped * 86400.0, np.degrees(amp_b_damped), color = color3, marker = '+', label = 'Model B')
# plt.loglog(freq_a2_damped * 86400.0, np.degrees(amp_a2_damped), color = color2, marker = 'x', label = 'Model A2')
# plt.loglog(freq_a1 * 86400.0, np.degrees(amp_a1), color = color1, marker = '.', label = 'Model A1')
# plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
# plt.axvline(average_mean_motion * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axvline(2.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.axvline(3.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.grid()
# plt.legend(loc = 'lower left')
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel('A [º]')
# plt.title('Frequency content of libration (damped dynamics)')



# read_dir = os.getcwd() + '/simulation-results/model-a1/2023-07-13 10:24:42.226993/'
# dependents = read_vector_history_from_file(read_dir + 'dependent-variable-history.dat')
# libration_translation = result2array(extract_elements_from_history(dependents, [4, 5]))
#
# read_dir = os.getcwd() + '/simulation-results/model-a2/2023-07-13 11:38:45.319336/'
# dependents = read_vector_history_from_file(read_dir + 'dependents-d8192.dat')
# libration_rotation = result2array(extract_elements_from_history(dependents, [4, 5]))
#
# read_dir = os.getcwd() + '/simulation-results/model-b/2023-07-13 12:35:19.273692/'
# dependents = read_vector_history_from_file(read_dir + 'dependents-d8192.dat')
# libration_coupled = result2array(extract_elements_from_history(dependents, [4, 5]))
#
# freq_trans, amp_trans = get_fourier(libration_translation, clean_signal = [TWOPI, 1])


# dependents = read_vector_history_from_file(read_dir + 'dependents-undamped.dat')
# libration_history = result2array(extract_elements_from_history(dependents, 5))
# freq, amp = get_fourier(libration_history, clean_signal = [TWOPI, 1])
# temp = amp[freq > 0.96*average_mean_motion]
# aux = freq[freq > 0.96*average_mean_motion]
# temp = temp[aux < 1.04*average_mean_motion]
# libration_amplitudes[0] = max(temp)

# for idx, current_dissipation_time in enumerate(dissipation_times):
#
#     time_str = str(int(current_dissipation_time / 3600.0))
#     print('Processing damping time of ' + time_str + ' hours.')
#     dependents = read_vector_history_from_file(read_dir + 'dependents-d' + time_str + '-full.dat')
#     libration_history = result2array(extract_elements_from_history(dependents, 5))
#     freq, amp = get_fourier(libration_history, clean_signal=[TWOPI, 1])
#     temp = amp[freq > 0.96 * average_mean_motion]
#     aux = freq[freq > 0.96 * average_mean_motion]
#     temp = temp[aux < 1.04 * average_mean_motion]
#     libration_amplitudes[idx+1] = max(temp)
#
# plt.figure()
# plt.plot(iterations, np.degrees(libration_amplitudes), marker = '.')
# plt.xlabel('Iteration')
# plt.ylabel('A [º]')
# plt.grid()
# plt.title('Libration amplitude through the damping process')

# plt.figure()
# plt.plot(np.degrees(libration_coupled[:,2]), np.degrees(libration_coupled[:,1]), marker = 'x', color = color3, label = 'Model B')
# plt.plot(np.degrees(libration_rotation[:,2]), np.degrees(libration_rotation[:,1]), marker = '.', color = color2, label = 'Model A2')
# plt.plot(np.degrees(libration_translation[:,2]), np.degrees(libration_translation[:,1]), marker = '.', color = color1, label = 'Model A1')
# plt.grid()
# plt.legend()
# plt.xlabel('Longitude [º]')
# plt.ylabel('Latitude[º]')
# plt.title('Tidal libration')

# dir = 'simulation-results/model-a1/'
# big_time_step_history = read_vector_history_from_file(dir + '2023-07-23 11:42:54.947535/state-history.dat')
# small_time_step_history = read_vector_history_from_file(dir + '2023-07-23 11:54:41.670469/state-history.dat')
# errors = compare_results(small_time_step_history, big_time_step_history, list(big_time_step_history.keys()))
# errors_array = result2array(errors)
#
# epochs_array = errors_array[:,0]
# position_error_array = errors_array[:,1:4]
# # velocity_error_array = errors_array[:,4:]
# normed_position_errors = norm(position_error_array, axis = 1)
#
# plt.figure()
# plt.semilogy(epochs_array / 86400.0, normed_position_errors, marker = '.')
# plt.axhline(1e-3, label = 'Millimeters', ls = '--', color = 'k')
# plt.legend()
# plt.grid()
# plt.xlabel('Time since simulation start [days]')
# plt.ylabel(r'$|\vec\varepsilon| (t)$ [m]')

# rkdp_eph = read_vector_history_from_file('ephemeris/rkdp/translation-a.eph')
# rkf_eph = read_vector_history_from_file('ephemeris/translation-a.eph')
# diffs = compare_results(rkdp_eph, rkf_eph, list(rkf_eph.keys()))
# normed_differences_a = dict.fromkeys(list(diffs.keys()))
# for key in list(normed_differences_a.keys()):
#     normed_differences_a[key] = norm(diffs[key][:3])
#
# rkdp_eph = read_vector_history_from_file('ephemeris/rkdp/translation-b.eph')
# rkf_eph = read_vector_history_from_file('ephemeris/translation-b.eph')
# diffs = compare_results(rkdp_eph, rkf_eph, list(rkf_eph.keys()))
# normed_differences_b = dict.fromkeys(list(diffs.keys()))
# for key in list(normed_differences_b.keys()):
#     normed_differences_b[key] = norm(diffs[key][:3])
#
# plt.figure()
# plt.semilogy(np.array(list(normed_differences_a.keys())) / constants.JULIAN_YEAR, list(normed_differences_a.values()), label = 'Model A')
# plt.semilogy(np.array(list(normed_differences_b.keys())) / constants.JULIAN_YEAR, list(normed_differences_b.values()), label = 'Model B')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.title('Position differences between the use of an RKDP7(8) and an RKF10(12) integrator')

# model = 'B'
#
# bodies = get_solar_system(model)
# # DEFINE PROPAGATION
# initial_epoch = constants.JULIAN_DAY
# if model == 'A1':
#     initial_state = read_vector_history_from_file('ephemeris/translation-a.eph')[initial_epoch]
# else:
#     initial_state = np.concatenate((read_vector_history_from_file('ephemeris/translation-b.eph')[initial_epoch],
#                                     read_vector_history_from_file('ephemeris/rotation-b.eph')[initial_epoch]))
# simulation_time = 30.0*constants.JULIAN_DAY
# dependent_variables = get_list_of_dependent_variables(model, bodies)
# propagator_settings = get_propagator_settings(model, bodies, initial_epoch, initial_state, simulation_time, dependent_variables)
#
# # Setup parameters settings to propagate the state transition matrix
# parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, [initial_epoch])
# parameter_settings = parameter_settings + \
#                      [estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 0), (2, 2)])]
# if model == 'A1':
#     parameter_settings = parameter_settings + \
#                          [estimation_setup.parameter.scaled_longitude_libration_amplitude('Phobos')]
#
# # Create the parameters that will be estimated
# parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
# variational_equations_solver = numerical_simulation.create_variational_equations_solver(bodies, propagator_settings, parameters_to_estimate)
#
# if model == 'A1':
#     errors_file = 'benchmark_errors/2023-08-01 10:44:43.176090/rkf1210-dt300/errors.dat'
# else:
#     errors_file = 'benchmark_errors/2023-08-03 10:34:59.494772/rkf1210-dt300/errors.dat'
# initial_state_perturbance = read_vector_history_from_file(errors_file)[initial_epoch]
# libration_perturbance = 0.01 * 2.695220284671387
# c20_perturbance = 0.01 * (-0.029243)
# c22_perturbance = 0.01 * 0.015664
#
# final_state_perturbance_as_per_errors_file = read_vector_history_from_file(errors_file)[initial_epoch + simulation_time]
#
# final_state_perturbance_due_to_initial_state_perturbance = variational_equations_solver.state_transition_matrix_history[initial_epoch + simulation_time] @ initial_state_perturbance
#
# if model == 'A1':
#     extra = 1
# else:
#     extra = 0
# final_state_perturbance_due_to_c20 = variational_equations_solver.sensitivity_matrix_history[initial_epoch + simulation_time][:,0+extra] * c20_perturbance
# final_state_perturbance_due_to_c22 = variational_equations_solver.sensitivity_matrix_history[initial_epoch + simulation_time][:,1+extra] * c22_perturbance
#
# final_state_perturbance = final_state_perturbance_due_to_initial_state_perturbance + \
#                           final_state_perturbance_due_to_c20 + \
#                           final_state_perturbance_due_to_c22
#
# if model == 'A1':
#     final_state_perturbance_due_to_libration = variational_equations_solver.sensitivity_matrix_history[
#                                                    initial_epoch + simulation_time][:,0] * libration_perturbance
#     final_state_perturbance = final_state_perturbance + \
#                               final_state_perturbance_due_to_libration
#
# print('Initial state: ', norm(final_state_perturbance_due_to_initial_state_perturbance[:3]))
# if model == 'A1':
#     print('A:', norm(final_state_perturbance_due_to_libration[:3]))
# print('C20:', norm(final_state_perturbance_due_to_c20[:3]))
# print('C22:', norm(final_state_perturbance_due_to_c22[:3]))
# print('Total: ', norm(final_state_perturbance[:3]))
#
# if model == 'A1':
#     cosas = np.zeros([6, 3])
# else:
#     cosas = np.zeros([13,3])
# cosas[:,0] = final_state_perturbance_due_to_initial_state_perturbance
# cosas[:,1] = final_state_perturbance_as_per_errors_file
# cosas[:,2] = final_state_perturbance_due_to_initial_state_perturbance - final_state_perturbance_as_per_errors_file
#
# dependents_a = result2array(read_vector_history_from_file('ephemeris/associated-dependents/a1.dat'))
# dependents_b = result2array(read_vector_history_from_file('ephemeris/associated-dependents/b.dat'))
# lat_a = bring_inside_bounds(dependents_a[:,5], -PI/2, PI/2, include = 'upper')
# lon_a = bring_inside_bounds(dependents_a[:,6], -PI, PI, include = 'upper')
# lat_b = bring_inside_bounds(dependents_b[:,5], -PI/2, PI/2, include = 'upper')
# lon_b = bring_inside_bounds(dependents_b[:,6], -PI, PI, include = 'upper')
#
# plt.figure()
# plt.scatter(lon_b * 360.0 / TWOPI, lat_b * 360.0 / TWOPI, marker = '.', label = 'Model B', color = color2)
# plt.scatter(lon_a * 360.0 / TWOPI, lat_a * 360.0 / TWOPI, marker = '.', label = 'Model A', color = color1)
# plt.grid()
# plt.legend()
# plt.xlabel('LON [º]')
# plt.ylabel('LAT [º]')
# plt.title('Tidal libration')

uncoupled = read_vector_history_from_file('simulation-results/model-a1/2023-08-05 21:38:24.833011/state-history.dat')
coupled = read_vector_history_from_file('ephemeris/translation-b.eph')
uncoupled_new = read_vector_history_from_file('simulation-results/model-a1/2023-08-08 21:08:09.167299/state-history.dat')
coupled_new = read_vector_history_from_file('ephemeris/new/translation-b.eph')
diffs = result2array(compare_results(uncoupled, coupled, list(coupled.keys())))
diffs_new = result2array(compare_results(uncoupled_new, coupled_new, list(coupled_new.keys())))
normed_diffs = norm_rows(diffs[:,1:4])
normed_diffs_new = norm_rows(diffs_new[:,1:4])
integration_errors = result2array(read_vector_history_from_file('benchmark_errors/A1/2023-08-08 22:02:22.057801/rkdp-dt270/errors.dat'))
normed_integration_errors = norm_rows(integration_errors[:,1:4])

deltas = result2array(
    compare_results(array2result(np.concatenate((np.atleast_2d(diffs[:,0]).T, np.atleast_2d(normed_diffs).T), 1)),
                    array2result(np.concatenate((np.atleast_2d(diffs_new[:,0]).T, np.atleast_2d(normed_diffs_new).T), 1)),
                    diffs_new[:,0])
)

factor = constants.JULIAN_YEAR
# factor = constants.JULIAN_DAY

plt.figure()
# plt.semilogy(diffs[:,0] / factor, normed_diffs, label = r'RKF10(12), $\Delta t = 5$min')
plt.semilogy(diffs_new[:,0] / factor, normed_diffs_new, label = r'Coupling effects')
# plt.semilogy(deltas[:,0] / factor, abs(deltas[:,1]), label = r'Difference between RKDP7(8) and RKF10(12)')
plt.semilogy(integration_errors[:,0] / factor, normed_integration_errors, label = r'Integration error')
plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], size = 15)
if factor == constants.JULIAN_DAY:
    plt.xlim([-1.0, 30.0])
    plt.xticks(size=15)
else:
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size = 15)
plt.xlabel('Time since J2000 [years]')
plt.ylabel(r'$\Delta r$ [m]')
plt.grid()
plt.legend()
plt.title('Position difference due to couplings compared to integration errors')


print('DONE.')