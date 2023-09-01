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


def fourier_transform(time_history: np.ndarray, clean_signal: list = [0.0, 0]) -> tuple:

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
        raise TypeError('(fourier_transform): Invalid input. The second entry in clean_signal should be of type "int". A type ' + str(type(clean_signal[1])) + 'was provided.')
    if clean_signal[1] < 0:
        raise ValueError('(fourier_transform): Invalid input. The second entry in clean_signal cannot be negative. Current values is ' + str(clean_signal[1]) + '.')
    if clean_signal[0] < 0.0:
        raise ValueError('(fourier_transform): Invalid input. The first entry in clean_signal cannot be negative. Current values is ' + str(clean_signal[1]) + '.')

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

# average_mean_motion = 0.00022785636553897436
# bodies = get_solar_system('A2', 'ephemeris/translation-c.eph', 'ephemeris/rotation-c.eph')
# normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor, average_mean_motion)
# # R = MarsEquatorOfDate(bodies).j2000_to_mars_rotation
# # full_state_rotation = np.concatenate((np.concatenate((R, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), R), 1)), 0)
# # mu_mars = bodies.get('Mars').gravitational_parameter
#
# # trajectory = read_vector_history_from_file('ephemeris/new/translation-b.eph')
# dependents = read_vector_history_from_file('ephemeris/new/associated-dependents/b.dat')
# dependents_array = dict2array(dependents)
#
# # reduced_dependents_array = dependents_array[dependents_array[:,0] <= 86400.0 * 30.0]
# reduced_dependents_array = dependents_array[dependents_array[:,0] <= dependents_array[-1,0]]

####################################################################################################################################
# phi = reduced_dependents_array[:,[0,3]]
# phi[:,1] = remove_jumps(phi[:,1], TWOPI)
# coeffs = np.polyfit(phi[:,0], phi[:,1], 1)
# physical_libration = phi.copy()
# physical_libration[:,1] = physical_libration[:,1] - (coeffs[1] + coeffs[0] * phi[:,0])
# physical_libration_from_euler_angle = physical_libration.copy()
# tidal_libration = reduced_dependents_array[:,[0,6]]
# raan = reduced_dependents_array[:,[0,9]]
# coeffs = np.polyfit(raan[:,0], raan[:,1], 1)
# non_secular_raan = raan.copy()
# non_secular_raan[:,1] = non_secular_raan[:,1] - (coeffs[1] + coeffs[0]*non_secular_raan[:,0])
# clean_physical_libration = physical_libration.copy()
# clean_physical_libration[:,1] = physical_libration[:,1] + non_secular_raan[:,1]
#
# physical_libration_frequency, physical_libration_amplitude = fourier_transform(physical_libration)
# clean_physical_libration_frequency, clean_physical_libration_amplitude = fourier_transform(clean_physical_libration)
# tidal_libration_frequency, tidal_libration_amplitude = fourier_transform(tidal_libration)
# raan_frequency, raan_amplitude = fourier_transform(raan)
# non_secular_raan_frequency, non_secular_raan_amplitude = fourier_transform(non_secular_raan)
################################################################################################################################
# continuous_phi = remove_jumps(phi, TWOPI)
# coeffs = np.polyfit(reduced_dependents_array[:,0], continuous_phi, 2)
# res = continuous_phi - (coeffs[2] + coeffs[1] * reduced_dependents_array[:,0] + coeffs[0]*reduced_dependents_array[:,0]**2)
# residual_history = np.zeros([len(reduced_dependents_array), 2])
# residual_history[:,0] = reduced_dependents_array[:,0]
# residual_history[:,1] = res
# residual_frequency, residual_amplitude = fourier_transform(residual_history)
# fourier_reconstruction = np.zeros(len(residual_history))
# for k in range(len(residual_frequency)):
#     fourier_reconstruction = fourier_reconstruction + residual_amplitude[k]*np.sin(residual_frequency[k]*reduced_dependents_array[:,0])

# plt.figure()
# plt.plot(reduced_dependents_array[:,0] / 86400.0, phi * 360.0 / TWOPI, marker = '.', label = 'Before remove jumps')
# plt.plot(reduced_dependents_array[:,0] / 86400.0, continuous_phi * 360.0 / TWOPI, marker = '.', label = 'After remove jumps')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\phi$ [º]')
# plt.title('Third Euler angle')
#
# plt.figure()
# plt.plot(reduced_dependents_array[:,0] / 86400.0, res * 360.0 / TWOPI, marker = '.', label = 'Residual')
# plt.plot(reduced_dependents_array[:,0] / 86400.0, fourier_reconstruction * 360.0 / TWOPI, marker = '.', label = 'Fourier reconstruction')
# # plt.plot(reduced_dependents_array[:,0] / 86400.0, continuous_phi * 360.0 / TWOPI, marker = '.', label = 'After remove jumps')
# plt.grid()
# # plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\phi - (\phi_o + \dot{\phi}t)$ [º]')
# plt.title('Residuals after quadratic fit')
#
# plt.figure()
# plt.loglog(residual_frequency, residual_amplitude, marker = '.')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/s]')
# plt.ylabel(r'$A$ [rad]')
# plt.title('Residual frequency content')
#####################################################################################################################################################

# plt.figure()
# plt.plot(physical_libration[:,0] / 86400.0, physical_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'Physical libration, $\tau$')
# plt.plot(tidal_libration[:,0] / 86400.0, tidal_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'Tidal libration, $\psi$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Libration')
#
# plt.figure()
# plt.plot(physical_libration[:,0] / 86400.0, physical_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'Physical libration, $\tau$')
# plt.plot(tidal_libration[:,0] / 86400.0, tidal_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'Tidal libration, $\psi$')
# plt.plot(raan[:,0] / 86400.0, raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\Omega$')
# plt.plot(non_secular_raan[:,0] / 86400.0, non_secular_raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tilde{\Omega} = \Omega-\Omega_o-\dot{\Omega}t$')
# plt.plot(clean_physical_libration[:,0] / 86400.0, clean_physical_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tau - \tilde{\Omega}$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Libration')
#
# plt.figure()
# plt.plot(tidal_libration[:,0] / 86400.0, tidal_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'Tidal libration, $\psi$')
# plt.plot(clean_physical_libration[:,0] / 86400.0, clean_physical_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tau - \tilde{\Omega}$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Libration')
#
# plt.figure()
# plt.loglog(clean_physical_libration_frequency * 86400.0, clean_physical_libration_amplitude * 360.0 / TWOPI, marker = '.', label = r'$\tau + \tilde{\Omega}$')
# plt.loglog(physical_libration_frequency * 86400.0, physical_libration_amplitude * 360.0 / TWOPI, marker = '.', label = r'Physical libration, $\tau$')
# plt.loglog(tidal_libration_frequency * 86400.0, tidal_libration_amplitude * 360.0 / TWOPI, marker = '.', label = r'Tidal libration, $\psi$')
# # plt.loglog(raan_frequency * 86400.0, raan_amplitude * 360.0 / TWOPI, marker = '.', label = r'$\Omega$')
# # plt.loglog(non_secular_raan_frequency * 86400.0, non_secular_raan_amplitude * 360.0 / TWOPI, marker = '.', label = r'$\tilde{\Omega}$')
# plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
# plt.axvline(average_mean_motion * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axvline(2.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.axvline(3.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'$A$ [º]')
# plt.title('Libration frequency content')
#
# omega = reduced_dependents_array[:,[0,11]]
# phi_plus_raan = phi.copy()
# phi_plus_raan[:,1] = phi[:,1] + raan[:,1] + omega[:,1]
# coeffs = np.polyfit(phi_plus_raan[:,0], phi_plus_raan[:,1], 1)
# non_secular_angle = phi_plus_raan.copy()
# non_secular_angle[:,1] = non_secular_angle[:,1] - (coeffs[1] + coeffs[0]*non_secular_angle[:,0])
# plt.figure()
# plt.plot(phi[:,0] / 86400.0, phi[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\phi$')
# plt.plot(raan[:,0] / 86400.0, raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\Omega$')
# plt.plot(raan[:,0] / 86400.0, phi[:,1]+raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\phi + \Omega$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Libration')
#
# plt.figure()
# plt.plot(non_secular_angle[:,0] / 86400.0, non_secular_angle[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tilde{\phi}+\tilde{\Omega}+\tilde{\omega}$')
# # plt.plot(raan[:,0] / 86400.0, raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\Omega$')
# # plt.plot(raan[:,0] / 86400.0, phi[:,1]+raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\phi + \Omega$')
# plt.grid()
# # plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Libration')

##########################################################################################################################
# omega = reduced_dependents_array[:,[0,11]]
# theta = reduced_dependents_array[:,[0,12]]
# mean_anomaly = np.zeros([len(theta), 2])
# mean_anomaly[:,0] = theta[:,0]
# for k in range(len(mean_anomaly)):
#     mean_anomaly[k,1] = true_to_mean_anomaly(theta[k,1], reduced_dependents_array[k,8])
#
# continuous_omega = omega.copy()
# continuous_omega[:,1] = remove_jumps(omega[:,1], TWOPI)
# continuous_raan = raan.copy()
# continuous_raan[:,1] = remove_jumps(raan[:,1], TWOPI)
# continuous_mean_anomaly = mean_anomaly.copy()
# continuous_mean_anomaly[:,1] = remove_jumps(mean_anomaly[:,1], TWOPI)
# continuous_mean_longitude = continuous_mean_anomaly.copy()
# continuous_mean_longitude[:,1] = continuous_raan[:,1] + continuous_omega[:,1] + continuous_mean_anomaly[:,1]
#
# continuous_psi = reduced_dependents_array[:,[0,1]]
# continuous_psi[:,1] = remove_jumps(continuous_psi[:,1], TWOPI)
# continuous_phi = reduced_dependents_array[:,[0,3]]
# continuous_phi[:,1] = remove_jumps(continuous_phi[:,1], TWOPI)
# continuous_spin = continuous_psi.copy()
# continuous_spin[:,1] = continuous_psi[:,1] + continuous_phi[:,1]
#
# plt.figure()
# plt.plot(continuous_raan[:,0] / 86400.0, continuous_raan[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\Omega$')
# plt.plot(continuous_omega[:,0] / 86400.0, continuous_omega[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\omega$')
# plt.plot(continuous_mean_anomaly[:,0] / 86400.0, continuous_mean_anomaly[:,1] * 360.0 / TWOPI, marker = '.', label = r'$M$')
# plt.plot(continuous_mean_longitude[:,0] / 86400.0, continuous_mean_longitude[:,1] * 360.0 / TWOPI, marker = '.', label = r'$u = \Omega + \omega + M$')
# # plt.plot(mean_anomaly[:,0] / 86400.0, mean_anomaly[:,1] * 360.0 / TWOPI, marker = '.', label = r'$M$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Translation angles')
#
# plt.figure()
# plt.plot(continuous_psi[:,0] / 86400.0, continuous_psi[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\psi$')
# plt.plot(continuous_phi[:,0] / 86400.0, continuous_phi[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\phi$')
# plt.plot(continuous_spin[:,0] / 86400.0, continuous_spin[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\lambda = \psi + \phi$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Rotation angles')
#
# physical_libration = np.zeros([len(reduced_dependents_array), 2])
# physical_libration[:,0] = reduced_dependents_array[:,0]
# physical_libration[:,1] = continuous_spin[:,1] - continuous_mean_longitude[:,1]
# plt.figure()
# plt.plot(continuous_spin[:,0] / 86400.0, continuous_spin[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\lambda$')
# plt.plot(continuous_mean_longitude[:,0] / 86400.0, continuous_mean_longitude[:,1] * 360.0 / TWOPI, marker = '.', label = r'$u$')
# plt.plot(physical_libration[:,0] / 86400.0, physical_libration[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tau = \lambda - u$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Physical libration')
#
# coeffs = np.polyfit(physical_libration[:,0], physical_libration[:,1], 1)
# physical_libration_residual = np.zeros([len(reduced_dependents_array), 2])
# physical_libration_residual[:,0] = reduced_dependents_array[:,0]
# physical_libration_residual[:,1] = physical_libration[:,1] - (coeffs[1] + coeffs[0] * physical_libration[:,0])
# plt.figure()
# plt.plot(physical_libration_residual[:,0] / 86400.0, physical_libration_residual[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tau - \tau_o - \dot{\tau}t$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Physical libration')
#
# plt.figure()
# plt.plot(physical_libration_from_euler_angle[:,0] / 86400.0, physical_libration_from_euler_angle[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tau$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Physical libration?')
#
# tau_frequency, tau_amplitude = fourier_transform(physical_libration_residual)
# plt.figure()
# plt.loglog(tau_frequency * 86400.0, tau_amplitude * 360.0 / TWOPI, marker = '.')
# plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
# plt.axvline(average_mean_motion * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axvline(2.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.axvline(3.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.grid()
# # plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'$A$ [º]')
# plt.title('Libration frequency content')

###########################################################################################################################
# #
# omega = reduced_dependents_array[:,[0,11]]
# continuous_omega = omega.copy()
# continuous_omega[:,1] = remove_jumps(omega[:,1], TWOPI)
# # # test_freq, test_amp = fourier_transform(continuous_omega)
# # frequency = 0.007636490796485515 / 86400.0
# # phase = -0.6340673764456879
# # H = np.zeros([len(omega), 3])
# # for k in range(len(H)):
# #     H[k] = [1, continuous_omega[k,0], np.cos(frequency*continuous_omega[k,0] - phase)]
# #
# # params = np.linalg.solve(H.T @ H, H.T @ continuous_omega[:,1])
# #
# coeffs = np.polyfit(continuous_omega[:,0], continuous_omega[:,1], 1)
# non_secular_omega = omega.copy()
# non_secular_omega[:,1] = continuous_omega[:,1] - (coeffs[1] + coeffs[0]*continuous_omega[:,0])
# plt.figure()
# plt.plot(omega[:,0] / 86400.0, bring_inside_bounds(omega[:,1], -PI, PI, 'upper') * 360.0 / TWOPI, marker = '.', label = r'$\omega$')
# # plt.plot(continuous_omega[:,0] / 86400.0, continuous_omega[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\omega$ (unbounded)')
# plt.plot(non_secular_omega[:,0] / 86400.0, non_secular_omega[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tilde{\omega} = \omega - \omega_o - \dot{\omega}t$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\omega$ [º]')
# plt.title('Argument of periapsis')
#
# test_line = continuous_omega.copy()
# test_line[:,1] = params[0] + params[1] * test_line[:,0] + params[2] * np.cos(frequency*test_line[:,0]-phase)
# plt.figure()
# # plt.plot(omega[:,0] / 86400.0, (continuous_omega[:,1] - non_secular_omega[:,1]) * 360.0 / TWOPI, marker = '.', label = r'$\tilde{\omega}$')
# plt.plot(test_line[:,0] / 86400.0, test_line[:,1] * 360.0 / TWOPI, marker = '.', label = r'Test line')
# plt.plot(omega[:,0] / 86400.0, continuous_omega[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\omega$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\omega$ [º]')
# plt.title('Argument of periapsis')


# plot_kepler_elements(extract_elements_from_history(dependents, [6, 7, 8, 9, 10, 11]))


# kepler = extract_elements_from_history(dependents, [6, 7, 8, 9, 10, 11])
# read_dir = 'simulation-results/model-b/2023-07-13 12:35:19.273692/'
# trajectory = read_vector_history_from_file(read_dir + 'states-d8192.dat')
# trajectory = extract_elements_from_history(trajectory, [0, 1, 2, 3, 4, 5])
# kepler = read_vector_history_from_file(read_dir + 'dependents-d8192.dat')
# kepler = extract_elements_from_history(kepler, [6, 7, 8, 9, 10, 11])
# epochs_list = list(trajectory.keys())
# state_wrt_mars = np.zeros([len(epochs_list), 7])
# n = np.zeros([len(epochs_list), 2])
# inc_ecc = np.zeros([len(epochs_list), 3])
#
# for idx, epoch in enumerate(epochs_list):
#
#     state_wrt_mars[idx,0] = epoch
#     n[idx,0] = epoch
# #     inc_ecc[idx,0] = current_epoch
# #
#     state_wrt_mars[idx,1:] = full_state_rotation @ trajectory[epoch]
#     n[idx,1] = semi_major_axis_to_mean_motion(kepler[epoch][0], mu_mars)
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
#
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
#
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

# plt.figure()
# plt.plot(n[:,0] / 86400.0 / 365.25, n[:,1], marker = '.')
# plt.grid()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$n$ [rad/s]')
# plt.title('Mean motion')

# mean_motion = dict2array(mean_motion_history_from_keplerian_history(extract_elements_from_history(dependents, [6, 7, 8, 9, 10, 11]), mu_mars))
# keplerian_array = dict2array(extract_elements_from_history(dependents, [6, 7, 8, 9, 10, 11]))
# average_elements, trash = average_over_integer_number_of_orbits(keplerian_array[:,:-1], keplerian_array)
#
# eccentricity = 0.015034167790105173
# average_mean_motion = 0.00022785636553897436                                          # AQUI PASAN COSAS
#
# dependents_array = dict2array(dependents)                                           # ESTAMOS AQUIIIIIII!!!!!!
#
# tid_lib_freq, tid_lib_amp = fourier_transform(dependents_array[:,[0,6]])                    # HOLA BUYENASSSSS
#
#
# dependents_array = dependents_array[dependents_array[:,0] <= 86400.0 * 90.0]
# M0 = true_to_mean_anomaly(eccentricity, dependents_array[0,7])
# M = bring_inside_bounds(M0 + average_mean_motion*(dependents_array[:,0] - dependents_array[0,0]), -PI, PI, include = 'upper')
#
# physical_libration = bring_inside_bounds(dependents_array[:,3] + dependents_array[:,1] - average_elements[2] - average_elements[4] - M, -PI, PI, include = 'upper')
# tidal_libration = dependents_array[:,6]
# lainey_physical_libration = tidal_libration - 2*eccentricity*np.sin(M)
# jacobson_physical_libration = tidal_libration + 2*eccentricity*np.sin(M)
#
# plt.figure()
# plt.plot(dependents_array[:,0] / 86400.0, lainey_physical_libration * 360.0 / TWOPI, marker = '.', label = 'Lainey\'s physical libration')
# plt.plot(dependents_array[:,0] / 86400.0, jacobson_physical_libration * 360.0 / TWOPI, marker = '.', label = 'Jacobson\'s physical libration')
# plt.plot(dependents_array[:,0] / 86400.0, tidal_libration * 360.0 / TWOPI, marker = '.', label = 'Tidal libration')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Librations')
#
# plt.figure()
# plt.plot(dependents_array[:,0] / 86400.0, dependents_array[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\Psi$')
# plt.plot(dependents_array[:,0] / 86400.0, dependents_array[:,2] * 360.0 / TWOPI, marker = '.', label = r'$\varphi$')
# # plt.plot(dependents_array[:,0] / 86400.0, tidal_libration * 360.0 / TWOPI, marker = '.', label = 'Tidal libration')
# plt.grid()
# # plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Euler angles')

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
# freq, amp = fourier_transform(np.concatenate((np.atleast_2d(epochs_array).T, np.atleast_2d(sph[:,2]).T), axis = 1), [TWOPI, 1])
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
# lib_a1 = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_a1, amp_a1 = fourier_transform(lib_a1[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-a2/2023-07-13 11:38:45.319336/dependents-undamped.dat'
# lib_a2_undamped = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_a2_undamped, amp_a2_undamped = fourier_transform(lib_a2_undamped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-a2/2023-07-13 11:38:45.319336/dependents-d8192.dat'
# lib_a2_damped = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_a2_damped, amp_a2_damped = fourier_transform(lib_a2_damped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-b/2023-07-13 12:35:19.273692/dependents-undamped.dat'
# lib_b_undamped = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_b_undamped, amp_b_undamped = fourier_transform(lib_b_undamped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-b/2023-07-13 12:35:19.273692/dependents-d8192.dat'
# lib_b_damped = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_b_damped, amp_b_damped = fourier_transform(lib_b_damped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-c/2023-07-13 22:02:17.013120/dependents-undamped.dat'
# lib_c_undamped = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_c_undamped, amp_c_undamped = fourier_transform(lib_c_undamped[:,[0,2]], [TWOPI, 1])
#
# file = 'simulation-results/model-c/2023-07-13 22:02:17.013120/dependents-d8192.dat'
# lib_c_damped = dict2array(extract_elements_from_history(read_vector_history_from_file(file), [4, 5]))
# freq_c_damped, amp_c_damped = fourier_transform(lib_c_damped[:,[0,2]], [TWOPI, 1])
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
# libration_translation = dict2array(extract_elements_from_history(dependents, [4, 5]))
#
# read_dir = os.getcwd() + '/simulation-results/model-a2/2023-07-13 11:38:45.319336/'
# dependents = read_vector_history_from_file(read_dir + 'dependents-d8192.dat')
# libration_rotation = dict2array(extract_elements_from_history(dependents, [4, 5]))
#
# read_dir = os.getcwd() + '/simulation-results/model-b/2023-07-13 12:35:19.273692/'
# dependents = read_vector_history_from_file(read_dir + 'dependents-d8192.dat')
# libration_coupled = dict2array(extract_elements_from_history(dependents, [4, 5]))
#
# freq_trans, amp_trans = fourier_transform(libration_translation, clean_signal = [TWOPI, 1])

# dependents = read_vector_history_from_file(read_dir + 'dependents-undamped.dat')
# libration_history = dict2array(extract_elements_from_history(dependents, 5))
# freq, amp = fourier_transform(libration_history, clean_signal = [TWOPI, 1])
# temp = amp[freq > 0.96*average_mean_motion]
# aux = freq[freq > 0.96*average_mean_motion]
# temp = temp[aux < 1.04*average_mean_motion]
# libration_amplitudes[0] = max(temp)
#
# for idx, current_dissipation_time in enumerate(dissipation_times):
#
#     time_str = str(int(current_dissipation_time / 3600.0))
#     print('Processing damping time of ' + time_str + ' hours.')
#     dependents = read_vector_history_from_file(read_dir + 'dependents-d' + time_str + '-full.dat')
#     libration_history = dict2array(extract_elements_from_history(dependents, 5))
#     freq, amp = fourier_transform(libration_history, clean_signal=[TWOPI, 1])
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
#
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
# errors_array = dict2array(errors)
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
# dependents_a = dict2array(read_vector_history_from_file('ephemeris/associated-dependents/a1.dat'))
# dependents_b = dict2array(read_vector_history_from_file('ephemeris/associated-dependents/b.dat'))
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

# uncoupled = read_vector_history_from_file('simulation-results/model-a1/2023-08-05 21:38:24.833011/state-history.dat')
# coupled = read_vector_history_from_file('ephemeris/translation-b.eph')
# uncoupled_new = read_vector_history_from_file('simulation-results/model-a1/2023-08-08 21:08:09.167299/state-history.dat')
# coupled_new = read_vector_history_from_file('ephemeris/new/translation-b.eph')
# diffs = dict2array(compare_results(uncoupled, coupled, list(coupled.keys())))
# diffs_new = dict2array(compare_results(uncoupled_new, coupled_new, list(coupled_new.keys())))
# normed_diffs = norm_rows(diffs[:,1:4])
# normed_diffs_new = norm_rows(diffs_new[:,1:4])
# integration_errors = dict2array(read_vector_history_from_file('benchmark_errors/A1/2023-08-08 22:02:22.057801/rkdp-dt270/errors.dat'))
# normed_integration_errors = norm_rows(integration_errors[:,1:4])
#
# deltas = dict2array(
#     compare_results(array2dict(np.concatenate((np.atleast_2d(diffs[:,0]).T, np.atleast_2d(normed_diffs).T), 1)),
#                     array2dict(np.concatenate((np.atleast_2d(diffs_new[:,0]).T, np.atleast_2d(normed_diffs_new).T), 1)),
#                     diffs_new[:,0])
# )
#
# factor = constants.JULIAN_YEAR
# # factor = constants.JULIAN_DAY
#
# plt.figure()
# # plt.semilogy(diffs[:,0] / factor, normed_diffs, label = r'RKF10(12), $\Delta t = 5$min')
# plt.semilogy(diffs_new[:,0] / factor, normed_diffs_new, label = r'Coupling effects')
# # plt.semilogy(deltas[:,0] / factor, abs(deltas[:,1]), label = r'Difference between RKDP7(8) and RKF10(12)')
# plt.semilogy(integration_errors[:,0] / factor, normed_integration_errors, label = r'Integration error')
# plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], size = 15)
# if factor == constants.JULIAN_DAY:
#     plt.xlim([-1.0, 30.0])
#     plt.xticks(size=15)
# else:
#     plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size = 15)
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.grid()
# plt.legend()
# plt.title('Position difference due to couplings compared to integration errors')

#####################################################################################################################################################

# r0 = 9484147.608363898
# v0 = 2112.9812867643805
# true_libration_amplitude = 2.695220284671387
# true_c20 = -0.029243
# true_c22 = 0.015664
#
# norm_residuals = True
# residuals_to_rsw = True
# length_of_estimated_state_vector = 6
# cols = int(20 + 4*(norm_residuals + 3*residuals_to_rsw) + length_of_estimated_state_vector + 1)
# data_matrix_base = read_matrix_from_file('estimation-results/batch 2023-08-12 10:25:25.573262/batch_analysis_matrix.dat', [9, cols])
# data_matrix_libration = read_matrix_from_file('estimation-results/bravo/batch 2023-08-12 10:55:22.243514/batch_analysis_matrix.dat', [9, cols+1])
# data_matrix_harmonics = read_matrix_from_file('estimation-results/alpha/batch 2023-08-12 11:44:38.653334/batch_analysis_matrix.dat', [8, cols + 2])
#
# extra_norm_res = 4
# extra_rsw_res = 12
# extra_rot = 0
#
# idx1 = 21 + extra_norm_res + extra_rsw_res
# idx2 = 27 + extra_norm_res + extra_rsw_res
# libration_to_base = data_matrix_libration[:,idx1:idx2] / data_matrix_base[:,idx1:idx2]
# harmonics_to_base = data_matrix_harmonics[:,idx1:idx2] / data_matrix_base[:-1,idx1:idx2]
#
# ms = 15
# xlabel = 'Duration of estimated arc [days]'
#
# plt.figure()
# plt.plot(data_matrix_base[:,0] / 86400.0, 100.0*abs(data_matrix_base[:,idx1]) / r0, marker='.', markersize=ms, label=r'$\Delta R_o$')
# plt.plot(data_matrix_base[:,0] / 86400.0, 100.0*abs(data_matrix_base[:,idx1+1]) / r0, marker='.', markersize=ms, label=r'$\Delta S_o$')
# plt.plot(data_matrix_base[:,0] / 86400.0, 100.0*abs(data_matrix_base[:,idx1+2]) / r0, marker='.', markersize=ms, label=r'$\Delta W_o$')
# plt.plot(data_matrix_base[:,0] / 86400.0, 100.0*abs(data_matrix_base[:,idx1+3]) / v0, marker='x', ls='dashed', label=r'$\Delta v_{r,o}$')
# plt.plot(data_matrix_base[:,0] / 86400.0, 100.0*abs(data_matrix_base[:,idx1+4]) / v0, marker='x', ls='dashed', label=r'$\Delta v_{s,o}$')
# plt.plot(data_matrix_base[:,0] / 86400.0, 100.0*abs(data_matrix_base[:,idx1+5]) / v0, marker='x', ls='dashed', label=r'$\Delta v_{w,o}$')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'$\Delta\vec x_o$ [% of $|\vec r_o|$ and $|\vec v_o|$]')
# plt.title('Error in estimated initial state w.r.t. truth for base estimation')
#
# plt.figure()
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_base[:,21], marker='.', label=r'$R$')
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_base[:,25], marker='.', label=r'$S$')
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_base[:,29], marker='.', label=r'$W$')
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_base[:,17], marker='.', label=r'$norm$')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'RMS($\varepsilon$) [m]')
# plt.title('Post-fit residual RMS')
#
# plt.figure()
# plt.plot(data_matrix_base[:,0] / 86400.0, abs(libration_to_base[:,0]), marker='.', markersize=ms, label=r'$\Delta R_o$')
# plt.plot(data_matrix_base[:,0] / 86400.0, abs(libration_to_base[:,1]), marker='.', markersize=ms, label=r'$\Delta S_o$')
# plt.plot(data_matrix_base[:,0] / 86400.0, abs(libration_to_base[:,2]), marker='.', markersize=ms, label=r'$\Delta W_o$')
# plt.plot(data_matrix_base[:,0] / 86400.0, abs(libration_to_base[:,3]), marker='x', ls='dashed', label=r'$\Delta v_{r,o}$')
# plt.plot(data_matrix_base[:,0] / 86400.0, abs(libration_to_base[:,4]), marker='x', ls='dashed', label=r'$\Delta v_{s,o}$')
# plt.plot(data_matrix_base[:,0] / 86400.0, abs(libration_to_base[:,5]), marker='x', ls='dashed', label=r'$\Delta v_{w,o}$')
# plt.plot(data_matrix_base[:,0] / 86400.0, 100*abs(data_matrix_libration[:,-1] / true_libration_amplitude), marker = '+', markersize=ms, label=r'$\Delta A$ [% of truth]')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'$\Delta\vec x_{o,bv} / \Delta\vec x_{o,bs}$')
# plt.title('Ratio of ERE between bravo and base estimations')
#
# plt.figure()
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, abs(harmonics_to_base[:,0]), marker='.', markersize=ms, label=r'$\Delta R_o$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, abs(harmonics_to_base[:,1]), marker='.', markersize=ms, label=r'$\Delta S_o$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, abs(harmonics_to_base[:,2]), marker='.', markersize=ms, label=r'$\Delta W_o$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, abs(harmonics_to_base[:,3]), marker='x', ls='dashed', label=r'$\Delta v_{r,o}$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, abs(harmonics_to_base[:,4]), marker='x', ls='dashed', label=r'$\Delta v_{s,o}$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, abs(harmonics_to_base[:,5]), marker='x', ls='dashed', label=r'$\Delta v_{w,o}$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, 100*abs(data_matrix_harmonics[:,-2] / true_c20), marker = '+', markersize=ms, label=r'$\Delta C_{2,0}$ [% of truth]')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, 100*abs(data_matrix_harmonics[:,-1] / true_c22), marker = '+', markersize=ms, label=r'$\Delta C_{2,2}$ [% of truth]')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'$\Delta\vec x_{o,\alpha} / \Delta\vec x_{o,bs}$')
# plt.title('Ratio of ERE between alpha and base estimations')
#
# plt.figure()
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_libration[:,21] / data_matrix_base[:,21], marker='.', label=r'$R$')
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_libration[:,25] / data_matrix_base[:,25], marker='.', label=r'$S$')
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_libration[:,29] / data_matrix_base[:,29], marker='.', label=r'$W$')
# plt.plot(data_matrix_base[:,0] / 86400.0, data_matrix_libration[:,17] / data_matrix_base[:,17], marker='.', label=r'$norm$')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'RMS($\varepsilon_{bv}$) / RMS($\varepsilon_{bs}$)')
# plt.title('Ratio of post-fit residual RMS between bravo and base estimations')
#
# plt.figure()
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, data_matrix_harmonics[:,21] / data_matrix_base[:-1,21], marker='.', label=r'$R$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, data_matrix_harmonics[:,25] / data_matrix_base[:-1,25], marker='.', label=r'$S$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, data_matrix_harmonics[:,29] / data_matrix_base[:-1,29], marker='.', label=r'$W$')
# plt.plot(data_matrix_harmonics[:,0] / 86400.0, data_matrix_harmonics[:,17] / data_matrix_base[:-1,17], marker='.', label=r'$norm$')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'RMS($\varepsilon_{\alpha}$) / RMS($\varepsilon_{bs}$)')
# plt.title('Ratio of post-fit residual RMS between alpha and base estimations')

# average_mean_motion = 0.0002278563609852602
# eccentricity = 0.015034167790105173

# batch = 'batch 2023-08-12 10:25:25.573262'
# estimation_time = '30'
# residuals = dict2array(read_vector_history_from_file('estimation-results/' + batch + '/estimation-time-' + estimation_time + '/residual-history-rsw.dat'))
# dependents = dict2array(compare_results(extract_elements_from_history(read_vector_history_from_file('ephemeris/new/associated-dependents/b.dat'), list(range(6,12))),
#                                           extract_elements_from_history(read_vector_history_from_file('ephemeris/new/associated-dependents/a1.dat'), list(range(6,12))),
#                                           residuals[:,0]))
#
# r_freq, r_amp = fourier_transform(residuals[:,[0,1]], [0.0, 1])
# s_freq, s_amp = fourier_transform(residuals[:,[0,2]], [0.0, 1])
# w_freq, w_amp = fourier_transform(residuals[:,[0,3]], [0.0, 1])
# a_freq, a_amp = fourier_transform(dependents[:,[0,1]], [0.0, 1])
# e_freq, e_amp = fourier_transform(dependents[:,[0,2]], [0.0, 1])
# i_freq, i_amp = fourier_transform(dependents[:,[0,3]], [TWOPI, 1])
# omega_freq, omega_amp = fourier_transform(dependents[:,[0,4]], [TWOPI, 1])
# raan_freq, raan_amp = fourier_transform(dependents[:,[0,5]], [TWOPI, 1])
# theta_freq, theta_amp = fourier_transform(dependents[:,[0,6]], [TWOPI, 1])
#
# plt.figure()
# plt.plot(r_freq * 86400.0, r_amp, marker='.', label=r'$R$')
# plt.plot(s_freq * 86400.0, s_amp, marker='.', label=r'$S$')
# plt.plot(w_freq * 86400.0, w_amp, marker='.', label=r'$W$')
# plt.plot(a_freq * 86400.0, a_amp, marker='.', ls = '--', label=r'$\Delta a$')
# plt.plot(e_freq * 86400.0, e_amp, marker='.', ls = '--', label=r'$\Delta e$ [-]')
# plt.plot(i_freq * 86400.0, i_amp * 360.0 / TWOPI, marker='.', ls = '--', label=r'$\Delta i$ [º]')
# plt.plot(raan_freq * 86400.0, raan_amp * 360.0 / TWOPI, marker='.', ls = '--', label=r'$\Delta\Omega$ [º]')
# plt.plot(omega_freq * 86400.0, omega_amp * 360.0 / TWOPI, marker='.', ls = '--', label=r'$\Delta\omega$ [º]')
# plt.plot(theta_freq * 86400.0, theta_amp * 360.0 / TWOPI, marker='.', ls = '--', label=r'$\Delta\theta$ [º]')
# plt.axvline(average_mean_motion * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axvline(2.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.axvline(3.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.yscale('log')
# plt.xscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'$A$ [m]')
# plt.title('Residual frequency content')

# mars_mu = 42828375815756.1
# initial_state = read_vector_history_from_file('ephemeris/new/translation-b.eph')
# dependents_coupled = dict2array(read_vector_history_from_file('ephemeris/new/associated-dependents/b.dat'))
# dependents_uncoupled = dict2array(read_vector_history_from_file('ephemeris/new/associated-dependents/a1.dat'))
#
# diffs = np.zeros([len(dependents_coupled), 7])
# diffs[:,0] = dependents_uncoupled[:,0]
# diffs[:,1:] = dependents_coupled[:,7:13] - dependents_uncoupled[:,7:13]
# diffs[:,4] = bring_inside_bounds(diffs[:,4], -PI, PI, include = 'upper')
# diffs[:,6] = bring_inside_bounds(diffs[:,6], -PI, PI, include = 'upper')
# diffs = array2dict(diffs)
#
# plot_kepler_elements(diffs, 'Differences in keplerian elements between coupled and uncoupled models')

# physical_libration = dependents_coupled[:,[0,3]]
# tidal_libration = dependents_coupled[:,[0,6]]
# tid_lib_freq, tid_lib_amp = fourier_transform(tidal_libration, [TWOPI, 1])
# phy_lib_freq, phy_lib_amp = fourier_transform(physical_libration, [TWOPI, 1])
#
# plt.figure()
# plt.loglog(tid_lib_freq * 86400.0, tid_lib_amp * 360.0 / TWOPI, marker = '.', label = 'Tidal libration')
# plt.loglog(phy_lib_freq * 86400.0, phy_lib_amp * 360.0 / TWOPI, marker = '.', label = 'Physical libration')
# plt.axvline(average_mean_motion * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
# plt.axvline(2.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# plt.axvline(3.0 * average_mean_motion * 86400.0, ls='dashed', c='r')
# # plt.loglog(tid_lib_freq * 86400.0, (tid_lib_amp + 2*eccentricity) * 360.0 / TWOPI, marker = '.', label = 'Physical libration')
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel('Amplitude [º]')
# plt.title('Tidal libration frequency content')
#
# bodies = get_solar_system('A2', 'ephemeris/translation-c.eph', 'ephemeris/rotation-c.eph')
# g3 = (get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor, 1.0)) ** 2
#
# temp = tid_lib_amp[tid_lib_freq > 0.96*average_mean_motion]
# aux = tid_lib_freq[tid_lib_freq > 0.96*average_mean_motion]
# temp = temp[aux < 1.04*average_mean_motion]
#
# tidal_libration_amplitude = max(temp)
#
# temp = phy_lib_amp[phy_lib_freq > 0.96*average_mean_motion]
# aux = phy_lib_freq[phy_lib_freq > 0.96*average_mean_motion]
# temp = temp[aux < 1.04*average_mean_motion]
#
# physical_libration_amplitude = max(temp)
#
# print('Scaled difference between physical and tidal libration amplitudes (it should be 2.0).')
# print((physical_libration_amplitude - tidal_libration_amplitude) / eccentricity)
#
# temp = phy_lib_amp[phy_lib_freq > 2*0.96*average_mean_motion]
# aux = phy_lib_freq[phy_lib_freq > 2*0.96*average_mean_motion]
# temp = temp[aux < 2*1.04*average_mean_motion]
#
# twice_per_orbit_libration_amplitude = max(temp)
#
# analytical_physical_libration_amplitude = 2*eccentricity / (1 - 1 / g3)
# analytical_twice_per_orbit_libration_amplitude = - eccentricity / (1 - 4 / g3) * (1.5*analytical_physical_libration_amplitude-17/4*eccentricity)

# kepler = cartesian_to_keplerian_history(initial_state, mars_mu)
# M = np.zeros([len(kepler), 2])
# M[:,0] = list(kepler.keys())
# for idx, epoch in enumerate(list(kepler.keys())):
#     M[idx,1] = true_to_mean_anomaly(kepler[epoch][1], kepler[epoch][-1])
# # M0 = true_to_mean_anomaly(kepler[1], kepler[-1])
# #
# # M = bring_inside_bounds(M0 + average_mean_motion*(dependents_coupled[:,0] - dependents_coupled[0,0]), -PI, PI, include = 'upper')
# physical_libration = bring_inside_bounds(dependents_coupled[:,3] - M[:,1], 0.0, TWOPI)
# tidal_libration = dependents_coupled[:,6]
#
# diff = physical_libration - tidal_libration
#
# plt.figure()
# plt.plot(dependents_coupled[:,0] / 86400.0 / 365.25, physical_libration * 360.0 / TWOPI, marker = '.', label = 'Physical libration')
# plt.plot(dependents_coupled[:,0] / 86400.0 / 365.25, tidal_libration * 360.0 / TWOPI, marker = '.', label = 'Tidal libration')
# # plt.plot(dependents_coupled[:,0] / 86400.0 / 365.25, diff * 360.0 / TWOPI, marker = '.', label = 'Optical libration')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\alpha$ [º]')
# plt.title('Librations')

save_dir = os.getcwd() + '/checking-partials/august-crisis/'
os.makedirs(save_dir, exist_ok = True)

increments = [-0.001, 0.00, 0.001]
initial_state = read_vector_history_from_file('ephemeris/translation-b.eph')[0.0]
simulation_time = 30.0 * constants.JULIAN_DAY

for idx, current_increment in enumerate(increments):

    current_libration_amplitude = (1.00 + current_increment) * 2.6952203863816266
    bodies = get_solar_system('A1', libration_amplitude=current_libration_amplitude)
    propagator_settings = get_propagator_settings('A1', bodies, 0.0, initial_state, simulation_time)

    if current_increment == 0.0:
        parameter_settings = (estimation_setup.parameter.initial_states(propagator_settings, bodies, [0.0]) +
                              [estimation_setup.parameter.scaled_longitude_libration_amplitude('Phobos')])
        parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
        simulator = numerical_simulation.create_variational_equations_solver(bodies,
                                                                             propagator_settings,
                                                                             parameters_to_estimate)
        write_matrix_history_to_file(simulator.sensitivity_matrix_history, save_dir + 'sensitivity-matrix-history.dat')

        for increment in [x for x in increments if x != 0.00]:
            analytical_delta = dict.fromkeys(list(simulator.state_history.keys()))
            for epoch in list(analytical_delta.keys()):
                analytical_delta[epoch] = (simulator.sensitivity_matrix_history[epoch] @ np.array([[increment]])).reshape(6)

            save2txt(analytical_delta, save_dir + 'analytical-delta' + str(int(np.sign(increment))) + '.dat')
    else:
        simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

    save2txt(simulator.state_history, save_dir + 'state-history-libration-index-' + str(idx) + '.dat')

analytical_delta_plus = read_vector_history_from_file(save_dir + 'analytical-delta1.dat')
analytical_delta_minus = read_vector_history_from_file(save_dir + 'analytical-delta-1.dat')
states_nominal = read_vector_history_from_file(save_dir + 'state-history-libration-index-1.dat')
states_plus = read_vector_history_from_file(save_dir + 'state-history-libration-index-2.dat')
states_minus = read_vector_history_from_file(save_dir + 'state-history-libration-index-0.dat')
epochs = list(states_nominal.keys())

numerical_delta_plus = compare_results(states_nominal, states_plus, epochs)
numerical_delta_minus = compare_results(states_nominal, states_minus, epochs)

dr_plus = np.zeros([len(epochs), 2])
dr_minus = np.zeros([len(epochs), 2])
dr_plus[:,0] = epochs
dr_minus[:,0] = epochs
dr_plus[:,1] = norm_rows(dict2array(compare_results(analytical_delta_plus, numerical_delta_plus, epochs))[:,1:4])
dr_minus[:,1] = norm_rows(dict2array(compare_results(analytical_delta_minus, numerical_delta_minus, epochs))[:,1:4])

epochs = np.array(epochs)
plt.figure()
plt.plot(epochs / 86400.0, dr_plus[:,1], marker = '.', label = r'$\Delta r_{+}$')
# plt.plot(epochs / 86400.0, dr_minus[:,1], marker = '.', label = r'$\Delta r_{-}$')
# plt.yscale('log')
plt.grid()
# plt.legend()
plt.xlabel('Time since J2000 [days]')
plt.ylabel(r'$\Delta r$ [m]')
plt.title('Difference between analytical and numerical deviations')

print('PROGRAM COMPLETED SUCCESSFULLY')

