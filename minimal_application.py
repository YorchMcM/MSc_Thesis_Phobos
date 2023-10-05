'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 7h 40min
    · Velocity of Phobos : 3 km/s
    · Reference radius: 13 km
'''
import os
from copy import copy, deepcopy
from Auxiliaries import *
from tudatpy.kernel.astro.element_conversion import cartesian_to_spherical

# save_dir = os.getcwd() + '/initial-guess-analysis/'
color1, color2, color3, color4 = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7f7f7f', '#bcbd22', '#17becf']
# colors = ['#0072BD', '#D95319', 'k', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7f7f7f', '#bcbd22', '#17becf']

########################################################################################################################
#                                                                                                                      #
#                                                     LIBRATIONS I                                                     #
#                                                                                                                      #
########################################################################################################################

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

# kepler = extract_elements_from_history(dependents, [6, 7, 8, 9, 10, 11])
# plot_kepler_elements(kepler)

########################################################################################################################
#                                                                                                                      #
#                                         ORBIT AND ROTATION VERIFICATION                                              #
#                                                                                                                      #
########################################################################################################################

# cwd = os.getcwd() + '/'
# data_dir = cwd + 'simulation-results/model-b/2023-09-14 20:34:13.535931/'
# dissipation_times = np.array([4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0,   8192.0])
# bodies = get_solar_system('B', cwd + 'ephemeris/translation-b.eph', cwd + 'ephemeris/rotation-b.eph')
# inertia_tensor = bodies.get('Phobos').inertia_tensor
# # R = MarsEquatorOfDate(bodies).j2000_to_mars_rotation
#
# undamped_dependents = dict2array(read_vector_history_from_file(data_dir + 'dependents-undamped.dat'))
# damped_dependents = dict2array(read_vector_history_from_file(data_dir + 'dependents-d8192-full.dat'))
#
# mars_mu = bodies.get('Mars').gravitational_parameter
# mean_motion_history = damped_dependents[:,[0,7]]
# mean_motion_history[:,1] = np.sqrt(mars_mu / mean_motion_history[:,1] ** 3)
# mean_motion, trash = average_over_integer_number_of_orbits(mean_motion_history, damped_dependents[:,[0,7,8,9,10,11,12]])
# normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(inertia_tensor, mean_motion)
#
# undamped_librations = undamped_dependents[:,[0,5,6]]
# damped_librations = damped_dependents[:,[0,5,6]]
#
# undamped_fourier = fourier_transform(undamped_librations)
# damped_fourier = fourier_transform(damped_librations)

# plt.figure()
# plt.loglog(undamped_fourier[:,0] * 86400.0, undamped_fourier[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\varphi$')
# plt.loglog(undamped_fourier[:,0] * 86400.0, undamped_fourier[:,2] * 360.0 / TWOPI, marker = '.', label = r'$\lambda$')
# plt.axvline(normal_mode * 86400.0, ls = 'dashed', c = 'k', label = r'$\nu$', linewidth = 2)
# plt.axvline(mean_motion * 86400.0, ls='dashed', c='g', label = r'$n$', linewidth = 2)
# plt.axvline(2.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.axvline(3.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'Amplitude [º]')
# plt.title('Undamped librational spectrum')
#
# plt.figure()
# plt.loglog(damped_fourier[:,0] * 86400.0, damped_fourier[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\varphi$')
# plt.loglog(damped_fourier[:,0] * 86400.0, damped_fourier[:,2] * 360.0 / TWOPI, marker = '.', label = r'$\lambda$')
# plt.axvline(normal_mode * 86400.0, ls = 'dashed', c = 'k', label = r'$\nu$', linewidth = 2)
# plt.axvline(mean_motion * 86400.0, ls='dashed', c='g', label = r'$n$', linewidth = 2)
# plt.axvline(2.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.axvline(3.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'Amplitude [º]')
# plt.title('Damped librational spectrum')

# spectrum = np.zeros([len(undamped_fourier), 2+len(dissipation_times)])
# spectrum[:,:2] = undamped_fourier[:,[0,2]]
# for idx, time in enumerate(dissipation_times):
#     current_file = data_dir + 'dependents-d' + str(time).replace('.0', '') + '-full.dat'
#     temp = dict2array(read_vector_history_from_file(current_file))[:,[0,6]]
#     temp = fourier_transform(temp)
#     spectrum[:,2+idx] = temp[:,1]
#
# spectrum = reduce_rows(spectrum, 10)
# plt.figure()
# plt.loglog(spectrum[:,0] * 86400.0, spectrum[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\tau = \infty$')
# for idx in range(len(dissipation_times)):
#     if idx % 2 == 0:
#         plt.loglog(spectrum[:,0] * 86400.0, spectrum[:,2+idx] * 360.0 / TWOPI, marker = '.', label = r'$n = ' + str(idx+2) + '$')
# plt.loglog(spectrum[:,0] * 86400.0, spectrum[:,-1] * 360.0 / TWOPI, marker = '.', label = r'$n = 13$')
# plt.axvline(normal_mode * 86400.0, ls = 'dashed', c = 'k', label = r'$\nu$', linewidth = 2)
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'Amplitude [º]')
# plt.title(r'Librational spectrum for $\tau = 2^n$ h.')
#
# undamped_librations = reduce_rows(undamped_librations, 14)
# damped_librations = reduce_rows(damped_librations, 14)
# plt.figure()
# plt.scatter(undamped_librations[:,2] / TWOPI * 360.0, undamped_librations[:,1] / TWOPI * 360.0, label = 'Undamped')
# plt.scatter(damped_librations[:,2] / TWOPI * 360.0, damped_librations[:,1] / TWOPI * 360.0, label = 'Damped')
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\lambda$ [º]')
# plt.ylabel(r'$\varphi$ [º]')
# plt.title('Mars\' coordinates in the Phobian sky')
# plt.axis('equal')
#
# undamped_fourier = reduce_rows(undamped_fourier, 10)
# damped_fourier = reduce_rows(damped_fourier, 10)
#
# plt.figure()
# plt.loglog(undamped_fourier[:,0] * 86400.0, undamped_fourier[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\varphi$')
# plt.loglog(undamped_fourier[:,0] * 86400.0, undamped_fourier[:,2] * 360.0 / TWOPI, marker = '.', label = r'$\lambda$')
# plt.axvline(normal_mode * 86400.0, ls = 'dashed', c = 'k', label = r'$\nu$', linewidth = 2)
# plt.axvline(mean_motion * 86400.0, ls='dashed', c='g', label = r'$n$', linewidth = 2)
# plt.axvline(2.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.axvline(3.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'Amplitude [º]')
# plt.title('Undamped librational spectrum')
#
# plt.figure()
# plt.loglog(damped_fourier[:,0] * 86400.0, damped_fourier[:,1] * 360.0 / TWOPI, marker = '.', label = r'$\varphi$')
# plt.loglog(damped_fourier[:,0] * 86400.0, damped_fourier[:,2] * 360.0 / TWOPI, marker = '.', label = r'$\lambda$')
# plt.axvline(normal_mode * 86400.0, ls = 'dashed', c = 'k', label = r'$\nu$', linewidth = 2)
# plt.axvline(mean_motion * 86400.0, ls='dashed', c='g', label = r'$n$', linewidth = 2)
# plt.axvline(2.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.axvline(3.0 * mean_motion * 86400.0, ls='dashed', c='g', linewidth = 2)
# plt.grid()
# plt.legend()
# plt.xlabel(r'$\omega$ [rad/day]')
# plt.ylabel(r'Amplitude [º]')
# plt.title('Damped librational spectrum')

# epochs = states[:,0]

# plt.figure()
# plt.scatter(mars_location[:,2] / TWOPI * 360.0, mars_location[:,1] / TWOPI * 360.0)
# plt.grid()
# plt.xlabel(r'$\lambda$ [º]')
# plt.ylabel(r'$\varphi$ [º]')
# plt.title('Mars\' coordinates in the Phobian sky')
# plt.axis('equal')
# plt.ylim([-0.2, 0.2])

# state_wrt_mars = np.zeros([len(epochs), 7])
# for idx, epoch in enumerate(epochs):
#
#     state_wrt_mars[idx,0] = epoch
#
#     state_wrt_mars[idx,1:4] = R @ states[idx,1:4]
#     state_wrt_mars[idx,4:] = R @ states[idx,4:]
#
# theta = np.linspace(0.0, TWOPI, 1001)
# circular_trajectory = np.zeros([len(theta), 3])
# R = np.mean(np.linalg.norm(state_wrt_mars[:,1:4], axis = 1))
# for idx in range(len(theta)):
#     angle = theta[idx]
#     circular_trajectory[idx] = np.array([R*np.cos(angle), R*np.sin(angle), 0.0])
#
# R_mars = 3390e3
#
# scale = R_mars
# figure, axis = plt.subplots()
# axis.add_patch(plt.Circle((0, 0), R_mars / scale, color=color2))
# axis.plot(state_wrt_mars[:,1] / scale, state_wrt_mars[:,2] / scale, label = 'Real orbit', c = color3)
# axis.plot(circular_trajectory[:,0] / scale, circular_trajectory[:,1] / scale, label = 'Circular orbit', c = 'purple')
# axis.set_xlabel(r'$x$ [$R$]')
# axis.set_ylabel(r'$y$ [$R$]')
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
# axis.add_patch(plt.Circle((0, 0), R_mars / scale, color=color2))
# axis.plot(state_wrt_mars[:,2] / scale, state_wrt_mars[:,3] / scale, label = 'Real orbit', c = color3)
# axis.plot(circular_trajectory[:,1] / scale, circular_trajectory[:,2] / scale, label = 'Equatorial orbit', c = 'purple')
# axis.set_xlabel(r'$y$ [$R$]')
# axis.set_ylabel(r'$z$ [$R$]')
# plt.grid()
# plt.legend()
# axis.set_title('Orbit\'s side view')
# plt.axis('equal')

# plt.figure()
# plt.plot(epochs / constants.JULIAN_YEAR, (kepler[:,1] - 0.0151) * 100.0, label = r'$\Delta e$ [$\times10^{-2}$]')
# plt.plot(epochs / constants.JULIAN_YEAR, kepler[:,3] * 360.0 / TWOPI - 1.1, label = r'$\Delta i$ [º]')
# plt.grid()
# plt.legend(loc = 'upper right')
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel('Element')
# plt.title('Orbit\'s inclination and eccentricity')
#
# plt.figure()
# plt.plot(epochs / constants.JULIAN_YEAR, (kepler[:,0] - 9376e3) / 1e3)
# plt.grid()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\Delta a$ [km]')
# plt.title('Orbit\'s semi-major axis')

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

########################################################################################################################
#                                                                                                                      #
#                                                     COUPLINGS                                                        #
#                                                                                                                      #
########################################################################################################################

# uncoupled = read_vector_history_from_file('simulation-results/model-a1/2023-08-05 21:38:24.833011/state-history.dat')
# uncoupled = read_vector_history_from_file('ephemeris/rkf1210/translation-a.eph')
# coupled = read_vector_history_from_file('ephemeris/rkf1210/translation-b.eph')
# # uncoupled_new = read_vector_history_from_file('simulation-results/model-a1/2023-08-08 21:08:09.167299/state-history.dat')
# # coupled_new = read_vector_history_from_file('ephemeris/new/translation-b.eph')
# diffs = dict2array(compare_results(uncoupled, coupled, list(coupled.keys())))
# # diffs_new = dict2array(compare_results(uncoupled_new, coupled_new, list(coupled_new.keys())))
# normed_diffs = norm_rows(diffs[:,1:4])
# # normed_diffs_new = norm_rows(diffs_new[:,1:4])
# integration_errors = dict2array(read_vector_history_from_file('benchmark_errors/A1/2023-08-08 22:02:22.057801/rkdp-dt270/errors.dat'))
# normed_integration_errors = norm_rows(integration_errors[:,1:4])
#
# # deltas = dict2array(
# #     compare_results(array2dict(np.concatenate((np.atleast_2d(diffs[:,0]).T, np.atleast_2d(normed_diffs).T), 1)),
# #                     array2dict(np.concatenate((np.atleast_2d(diffs_new[:,0]).T, np.atleast_2d(normed_diffs_new).T), 1)),
# #                     diffs_new[:,0])
# # )
#
# factor = constants.JULIAN_YEAR
# # factor = constants.JULIAN_DAY
#
# before_a_month = diffs[:,0] <= 30.0*constants.JULIAN_DAY
# plt.figure()
# plt.semilogy(diffs[:,0] / factor, normed_diffs, label = r'Coupling effects')
# # plt.semilogy(diffs_new[:,0] / factor, normed_diffs_new, label = r'Coupling effects')
# # plt.semilogy(deltas[:,0] / factor, abs(deltas[:,1]), label = r'Difference between RKDP7(8) and RKF10(12)')
# plt.semilogy(integration_errors[:,0] / factor, normed_integration_errors, label = r'Integration error')
# # plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
# if factor == constants.JULIAN_DAY:
#     plt.xlim([-1.0, 30.0])
#     # plt.xticks(size=15)
# else:
#     plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.grid()
# plt.legend()
# plt.title('Position difference due to couplings')

########################################################################################################################
#                                                                                                                      #
#                                              ESTIMATIONS COMPARISON                                                  #
#                                                                                                                      #
########################################################################################################################

# r0 = 9484147.608363898
# v0 = 2112.9812867643805
# true_libration_amplitude = 2.695220284671387
# true_c20 = -0.029243
# true_c22 = 0.015664
#
# norm_residuals = True
# residuals_to_rsw = True
# length_of_estimated_state_vector = 6
# cols = int(20 + 4*(norm_residuals + 3*residuals_to_rsw) + length_of_estimated_state_vector)
# data_matrix_base = read_matrix_from_file('estimation-results/base-1/batch 2023-09-15 11:48:08.314339/batch_analysis_matrix.dat', [9, cols])
# data_matrix_libration = read_matrix_from_file('estimation-results/bravo/batch 2023-09-16 08:14:54.184457/batch_analysis_matrix.dat', [9, cols+1])
# # data_matrix_harmonics = read_matrix_from_file('estimation-results/alpha/batch 2023-08-12 11:44:38.653334/batch_analysis_matrix.dat', [8, cols + 2])
#
# extra_norm_res = 4
# extra_rsw_res = 12
# extra_rot = 0
#
# idx1 = 20 + extra_norm_res + extra_rsw_res
# idx2 = 26 + extra_norm_res + extra_rsw_res
# libration_to_base = data_matrix_libration[:,idx1:idx2] / data_matrix_base[:,idx1:idx2]
# # harmonics_to_base = data_matrix_harmonics[:,idx1:idx2] / data_matrix_base[:-1,idx1:idx2]
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
# # plt.plot(data_matrix_base[:,0] / 86400.0, 100*abs(data_matrix_libration[:,-1] / true_libration_amplitude), marker = '+', markersize=ms, label=r'$\Delta B$ [% of truth]')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel(xlabel)
# plt.ylabel(r'$\Delta\vec x_{o,bv} / \Delta\vec x_{o,bs}$')
# plt.title('ERE improvement between bravo and base')

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
# plt.title('Residual improvement between bravo and base')

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

########################################################################################################################
#                                                                                                                      #
#                                                ORBIT'S FREQUENCIES                                                   #
#                                                                                                                      #
########################################################################################################################

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

########################################################################################################################
#                                                                                                                      #
#                                                   LIBRATIONS I                                                       #
#                                                                                                                      #
########################################################################################################################

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

########################################################################################################################
#                                                                                                                      #
#                                                   LIBRATIONS II                                                      #
#                                                                                                                      #
########################################################################################################################

# bodies = get_solar_system('A1')

# dependents_dict = read_vector_history_from_file('/home/yorch/thesis/ephemeris/associated-dependents/b.dat')
# dependents = dict2array(dependents_dict)
# euler_angles = dependents[:,[0,1,2,3]]
# keplerian_history = dependents[:,[0,7,8,9,10,11,12]]
# tidal_libration = dependents[:,[0,6]]
# tidal_libration[:,1] = -tidal_libration[:,1]
# mean_anomaly = keplerian_history[:,[0,6]]
# mean_anomaly[:,1] = np.array([true_to_mean_anomaly(keplerian_history[idx,2], keplerian_history[idx,-1]) for idx in range(len(keplerian_history))])
# eccentricity = compute_eccentricity_from_dependent_variables(dependents_dict)
# optical_libration = dependents[:,[0,1]]
# optical_libration[:,1] = 2*eccentricity*np.sin(mean_anomaly[:,1])
#
# offset = 40
# plt.figure()
# plt.plot(tidal_libration[offset:offset+100,0] / 3600.0, tidal_libration[offset:offset+100,1] * 360.0 / TWOPI, marker = '.', label = r'$\psi$')
# plt.plot(tidal_libration[offset:offset+100,0] / 3600.0, 2*eccentricity*np.sin(mean_anomaly[offset:offset+100,1]) * 360.0 / TWOPI, marker = '.', label = r'$2e$sin$M$')
# plt.plot(tidal_libration[offset:offset+100,0] / 3600.0, (tidal_libration[offset:offset+100,1] + optical_libration[offset:offset+100,1]) * 360.0 / TWOPI, marker = '.', label = r'$\gamma$')
# plt.grid()
# plt.legend(loc = 'lower right')
# plt.xlabel('Time since J2000 [hours]')
# plt.ylabel('Angle [º]')
# plt.title('Librations')

########################################################################################################################
#                                                                                                                      #
#                                            ANALYTICAL/NUMERICAL DEVIATIONS                                           #
#                                                                                                                      #
########################################################################################################################

# increments = [-0.001, 0.00, 0.001]
# initial_state = read_vector_history_from_file('ephemeris/translation-b.eph')[0.0]
# simulation_time = 30.0 * constants.JULIAN_DAY
#
# for idx, current_increment in enumerate(increments):
#
#     current_libration_amplitude = (1.00 + current_increment) * 2.6952203863816266
#     bodies = get_solar_system('A1', libration_amplitude=current_libration_amplitude)
#     propagator_settings = get_propagator_settings('A1', bodies, 0.0, initial_state, simulation_time)
#
#     if current_increment == 0.0:
#         parameter_settings = (estimation_setup.parameter.initial_states(propagator_settings, bodies, [0.0]) +
#                               [estimation_setup.parameter.scaled_longitude_libration_amplitude('Phobos')])
#         parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
#         simulator = numerical_simulation.create_variational_equations_solver(bodies,
#                                                                              propagator_settings,
#                                                                              parameters_to_estimate)
#         write_matrix_history_to_file(simulator.sensitivity_matrix_history, save_dir + 'sensitivity-matrix-history.dat')
#
#         for increment in [x for x in increments if x != 0.00]:
#             analytical_delta = dict.fromkeys(list(simulator.state_history.keys()))
#             for epoch in list(analytical_delta.keys()):
#                 analytical_delta[epoch] = (simulator.sensitivity_matrix_history[epoch] @ np.array([[increment]])).reshape(6)
#
#             save2txt(analytical_delta, save_dir + 'analytical-delta' + str(int(np.sign(increment))) + '.dat')
#     else:
#         simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
#
#     save2txt(simulator.state_history, save_dir + 'state-history-libration-index-' + str(idx) + '.dat')
#
# analytical_delta_plus = read_vector_history_from_file(save_dir + 'analytical-delta1.dat')
# analytical_delta_minus = read_vector_history_from_file(save_dir + 'analytical-delta-1.dat')
# states_nominal = read_vector_history_from_file(save_dir + 'state-history-libration-index-1.dat')
# states_plus = read_vector_history_from_file(save_dir + 'state-history-libration-index-2.dat')
# states_minus = read_vector_history_from_file(save_dir + 'state-history-libration-index-0.dat')
# epochs = list(states_nominal.keys())
#
# numerical_delta_plus = compare_results(states_nominal, states_plus, epochs)
# numerical_delta_minus = compare_results(states_nominal, states_minus, epochs)
#
# dr_plus = np.zeros([len(epochs), 2])
# dr_minus = np.zeros([len(epochs), 2])
# dr_plus[:,0] = epochs
# dr_minus[:,0] = epochs
# dr_plus[:,1] = norm_rows(dict2array(compare_results(analytical_delta_plus, numerical_delta_plus, epochs))[:,1:4])
# dr_minus[:,1] = norm_rows(dict2array(compare_results(analytical_delta_minus, numerical_delta_minus, epochs))[:,1:4])
#
# epochs = np.array(epochs)
# plt.figure()
# plt.plot(epochs / 86400.0, dr_plus[:,1], marker = '.', label = r'$\Delta r_{+}$')
# # plt.plot(epochs / 86400.0, dr_minus[:,1], marker = '.', label = r'$\Delta r_{-}$')
# # plt.yscale('log')
# plt.grid()
# # plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.title('Difference between analytical and numerical deviations')

########################################################################################################################
#                                                                                                                      #
#                                            COUPLING EFFECTS ON STATE HISTORY                                         #
#                                                                                                                      #
########################################################################################################################

# uncoupled_states = dict2array(read_vector_history_from_file('ephemeris/translation-a.eph'))
# coupled_states = dict2array(read_vector_history_from_file('ephemeris/translation-b.eph'))
# diffs = coupled_states[:,1:4] - uncoupled_states[:,1:4]
#
# uncoupled_dependents = dict2array(read_vector_history_from_file('ephemeris/associated-dependents/a1.dat'))
# coupled_dependents = dict2array(read_vector_history_from_file('ephemeris/associated-dependents/b.dat'))
# kepler_diffs = coupled_dependents[:,[0,7,8,9,10,11,12]] - uncoupled_dependents[:,[0,7,8,9,10,11,12]]
# kepler_diffs[:,0] = coupled_dependents[:,0]
# kepler_diffs[:,[3,4,5,6]] = bring_inside_bounds(kepler_diffs[:,[3,4,5,6]], -PI, PI, include = 'upper')
#
# plot_kepler_elements(kepler_diffs, title = 'Differences')
#
# plt.figure()
# plt.plot(kepler_diffs[:,0] / constants.JULIAN_YEAR, kepler_diffs[:,5] * 360.0 / TWOPI * 1000.0, label = r'$\Delta\omega$')
# plt.plot(kepler_diffs[:,0] / constants.JULIAN_YEAR, kepler_diffs[:,6] * 360.0 / TWOPI * 1000.0, label = r'$\Delta\theta$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'[$\times 10^{-3}$ º]')
# plt.title('Effect of coupling')
#
# plt.figure()
# plt.plot(kepler_diffs[:,0] / constants.JULIAN_YEAR, kepler_diffs[:,1] / 1000.0, label = r'$\Delta a$ [km]')
# plt.plot(kepler_diffs[:,0] / constants.JULIAN_YEAR, kepler_diffs[:,4] * 360.0 / TWOPI, label = r'$\Delta i$ [º]')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [years]')
# plt.title('Effect of coupling')
#
# plt.figure()
# plt.plot(kepler_diffs[:,0] / constants.JULIAN_YEAR, kepler_diffs[:,2])
# plt.grid()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\Delta e$ [-]')
# plt.title('Effect of coupling')
#
# plt.figure()
# plt.plot(kepler_diffs[:,0] / constants.JULIAN_YEAR, kepler_diffs[:,3] / TWOPI * 360.0 * 1000.0)
# plt.grid()
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$\Delta\Omega$ [$\times 10^{-3}\ º$]')
# plt.title('Effect of coupling')
#
# for idx in range(len(uncoupled_states)):
#     diffs[idx] = inertial_to_rsw_rotation_matrix(uncoupled_states[idx,1:]) @ diffs[idx]
#
# first_week = uncoupled_states[:,0] <= 7.0*constants.JULIAN_DAY
# first_month = uncoupled_states[:,0] <= 30.0*constants.JULIAN_DAY
#
# plt.figure()
# plt.plot(coupled_states[:,0] / 86400.0 / 365.25, norm_rows(diffs), marker = '.')
# plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
# plt.grid()
# plt.yscale('log')
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$|\Delta\vec r|$ [m]')
# plt.title('Position difference due to couplings')
#
# plt.figure()
# plt.plot(coupled_states[first_month,0] / 86400.0, norm_rows(diffs[first_month,:]), marker = '.')
# plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
# plt.grid()
# plt.yscale('log')
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$|\Delta\vec r|$ [m]')
# plt.title('Position difference due to couplings')
#
# plt.figure()
# plt.plot(coupled_states[:,0] / 86400.0 / 365.25, abs(diffs[:,0]), label = 'R')
# plt.plot(coupled_states[:,0] / 86400.0 / 365.25, abs(diffs[:,1]), label = 'S')
# plt.plot(coupled_states[:,0] / 86400.0 / 365.25, abs(diffs[:,2]), label = 'W')
# plt.yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
# plt.grid()
# plt.legend()
# plt.yscale('log')
# plt.xlabel('Time since J2000 [years]')
# plt.ylabel(r'$|\Delta\vec r_i|$ [m]')
# plt.title('Position difference due to couplings')

# plt.figure()
# plt.scatter(-diffs[:,0], -diffs[:,1], c = uncoupled_states[:,0] / constants.JULIAN_YEAR)
# plt.grid()
# cb = plt.colorbar(label='Time [days]')
# cb.set_ticks(ticks=[0, 2, 4, 6, 8, 10], labels=[0, 2, 4, 6, 8, 10], fontsize=15)
# plt.xlabel(r'$R$ [m]')
# plt.ylabel(r'$S$ [m]')
# plt.title('Uncoupled orbit around coupled orbit')
#
# plt.figure()
# plt.scatter(-diffs[:,0], -diffs[:,2], c = uncoupled_states[:,0] / constants.JULIAN_YEAR)
# plt.grid()
# cb = plt.colorbar(label='Time [days]')
# cb.set_ticks(ticks=[0, 2, 4, 6, 8, 10], labels=[0, 2, 4, 6, 8, 10], fontsize=15)
# plt.xlabel(r'$R$ [m]')
# plt.ylabel(r'$W$ [m]')
# plt.title('Uncoupled orbit around coupled orbit')

# plt.figure()
# plt.scatter(-diffs[first_week,0], -diffs[first_week,1], c = uncoupled_states[first_week,0] / 86400.0)
# plt.grid()
# cb = plt.colorbar(label='Time [days]')
# cb.set_ticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7], labels=[0, 1, 2, 3, 4, 5, 6, 7], fontsize=15)
# plt.xlabel(r'$R$ [m]')
# plt.ylabel(r'$S$ [m]')
# plt.title('Uncoupled orbit around coupled orbit')
#
# plt.figure()
# plt.scatter(-diffs[first_week,0], -diffs[first_week,2], c = uncoupled_states[first_week,0] / 86400.0)
# plt.grid()
# cb = plt.colorbar(label='Time [days]')
# cb.set_ticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7], labels=[0, 1, 2, 3, 4, 5, 6, 7], fontsize=15)
# plt.xlabel(r'$R$ [m]')
# plt.ylabel(r'$W$ [m]')
# plt.title('Uncoupled orbit around coupled orbit')

########################################################################################################################
#                                                                                                                      #
#                                            ORBIT DETERMINATION VERIFICATION                                          #
#                                                                                                                      #
########################################################################################################################

# Analysis on steps for numerical derivatives

# settings = EstimationSettings(os.getcwd() + '/estimation-settings.inp')
# translational_ephemeris_file, rotational_ephemeris_file = retrieve_ephemeris_files('A1')
# bodies = get_solar_system('A1', translational_ephemeris_file, rotational_ephemeris_file)
#
# initial_epoch = 1.0*constants.JULIAN_YEAR
# simulation_time = 30.0*constants.JULIAN_DAY
# initial_state = read_vector_history_from_file(os.getcwd() + '/ephemeris/translation-a.eph')[initial_epoch]
#
# propagator_settings = get_propagator_settings('A1', bodies, initial_epoch, initial_state, simulation_time, time_step = 150.0)
# precise_states = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings).state_history
#
# propagator_settings = get_propagator_settings('A1', bodies, initial_epoch, initial_state, simulation_time)
# reference_states = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings).state_history
# epochs = np.array(list(reference_states.keys()))
# epochs = (epochs - epochs[0]) / 86400.0
#
# numerical_errors = dict2array(compare_results(precise_states, reference_states, list(reference_states.keys())))[:,[1,2,3]]
#
#
# # DERIVATIVES WRT INITIAL STATE VECTOR
# small_steps = np.array([1.0, 1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-6])
# big_steps = 10.0*small_steps
# derivatives_small = compute_numerical_partials_wrt_initial_state_vector(bodies, propagator_settings)
# derivatives_big = compute_numerical_partials_wrt_initial_state_vector(bodies, propagator_settings, big_steps)
# diffs = [None]*6
# physical_differences = [None]*6
# for idx in range(len(derivatives_small)):
#     diffs[idx] = dict2array(compare_results(derivatives_small[idx], derivatives_big[idx], list(derivatives_big[idx].keys())))[:,[1,2,3]]
#     derivatives_big[idx] = dict2array(derivatives_big[idx])[:,[1,2,3]]
#     derivatives_small[idx] = dict2array(derivatives_small[idx])[:,[1,2,3]]
#     physical_differences[idx] = derivatives_small[idx] * small_steps[idx]
#
# legends = [r'x_o', r'y_o', r'z_o', r'v_{x,o}', r'v_{y,o}', r'v_{z,o}']
# plt.figure()
# for idx in range(6):
#     if idx < 3:
#         unit = r'[-]'
#     else:
#         unit = r'[s]'
#     if idx == 1:
#         plt.plot(epochs, norm_rows(derivatives_big[idx]), c = colors[idx], label = r'$\partial\vec r / \partial ' + legends[idx] + r'$ ' + unit, linewidth = 5)
#         plt.plot(epochs, norm_rows(diffs[idx]), c = colors[idx], ls = 'dashed', linewidth = 5)
#     else:
#         plt.plot(epochs, norm_rows(derivatives_big[idx]), c = colors[idx], label = r'$\partial\vec r / \partial ' + legends[idx] + r'$ ' + unit)
#         plt.plot(epochs, norm_rows(diffs[idx]), c = colors[idx], ls = 'dashed')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since estimation start [days]')
# plt.ylabel(r'$|\vec f|$ or $|\Delta\vec f|$')
# plt.title(r'Central differences: values and errors')
#
# plt.figure()
# for idx in range(6):
#     if idx == 1:
#         plt.plot(epochs, norm_rows(physical_differences[idx]), c = colors[idx], linewidth = 5, label = r'$\Delta ' + legends[idx] + r'$')
#     else:
#         plt.plot(epochs, norm_rows(physical_differences[idx]), c=colors[idx], label=r'$\Delta ' + legends[idx] + r'$')
# plt.plot(epochs, norm_rows(numerical_errors),  c = 'r', label = 'Integration error')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since estimation start [days]')
# plt.ylabel(r'$|\Delta\vec r|$ [m]')
# plt.title(r'Effect of each $h_2$ for each component of $\vec x_o$')
#
# # DERIVATIVES WRT LIBRATION AMPLITUDE AND HARMONIC COEFFICIENTS
# derivatives_small = (compute_numerical_partials_wrt_scaled_libration_amplitude(bodies, initial_epoch, initial_state, simulation_time) +
#                      compute_numerical_partials_wrt_harmonic_coefficients(bodies, initial_epoch, initial_state, simulation_time))
# derivatives_big = (compute_numerical_partials_wrt_scaled_libration_amplitude(bodies, initial_epoch, initial_state, simulation_time, step = 0.01) +
#                    compute_numerical_partials_wrt_harmonic_coefficients(bodies, initial_epoch, initial_state, simulation_time, step_vector = np.array([0.01, 0.01])))
# diffs = [None]*3
# physical_differences = [None]*3
# for idx in range(len(derivatives_small)):
#     diffs[idx] = dict2array(compare_results(derivatives_small[idx], derivatives_big[idx], list(derivatives_big[idx].keys())))[:,[1,2,3]]
#     derivatives_big[idx] = dict2array(derivatives_big[idx])[:,[1,2,3]]
#     derivatives_small[idx] = dict2array(derivatives_small[idx])[:,[1,2,3]]
#     physical_differences[idx] = derivatives_small[idx] * small_steps[idx]
#
# legends = [r'B', r'C_{2,0}', r'C_{2,2}']
# plt.figure()
# for idx in range(3):
#     if idx < 3:
#         unit = r'[-]'
#     else:
#         unit = r'[s]'
#     if idx == 1:
#         plt.plot(epochs, norm_rows(derivatives_big[idx]), c = colors[idx], label = r'$\partial\vec r / \partial ' + legends[idx] + r'$ ' + unit, linewidth = 5)
#         plt.plot(epochs, norm_rows(diffs[idx]), c = colors[idx], ls = 'dashed', linewidth = 5)
#     else:
#         plt.plot(epochs, norm_rows(derivatives_big[idx]), c = colors[idx], label = r'$\partial\vec r / \partial ' + legends[idx] + r'$ ' + unit)
#         plt.plot(epochs, norm_rows(diffs[idx]), c = colors[idx], ls = 'dashed')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since estimation start [days]')
# plt.ylabel(r'$|\vec f|$ or $|\Delta\vec f|$')
# plt.title(r'Central differences: values and errors')
#
# plt.figure()
# for idx in range(3):
#     if idx == 1:
#         plt.plot(epochs, norm_rows(physical_differences[idx]), c = colors[idx], linewidth = 5, label = r'$\Delta ' + legends[idx] + r'$')
#     else:
#         plt.plot(epochs, norm_rows(physical_differences[idx]), c=colors[idx], label=r'$\Delta ' + legends[idx] + r'$')
# plt.plot(epochs, norm_rows(numerical_errors),  c = 'r', label = 'Integration error')
# plt.yscale('log')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since estimation start [days]')
# plt.ylabel(r'$|\Delta\vec r|$ [m]')
# plt.title(r'Effect of $h_2$ for each parameter')

########################################################################################################################
#                                                                                                                      #
#                                               OBSERVATION SIMULATORS                                                 #
#                                                                                                                      #
########################################################################################################################

# cwd = os.getcwd() + '/'

# base = os.getcwd() + '/estimation-results/verification/changed-observation-simulation/'
# mothered = read_vector_history_from_file(base + '2023-09-12 20:45:08.494528/observation-history.dat')
# rogue = read_vector_history_from_file(base + '2023-09-12 20:48:53.453504/observation-history.dat')
# observation_times = np.array(list(rogue.keys()))
# observation_times = ( observation_times - observation_times[0] + 3600.0 ) / 86400.0
#
# rogue_array = dict2array(rogue)[:,[1,2,3]]
# mothered_array = dict2array(mothered)[:,[1,2,3]]
# diffs = rogue_array - mothered_array
# plt.figure()
# plt.scatter(observation_times, abs(rogue_array[:,0]), label = 'Rogue simulators')
# plt.scatter(observation_times, abs(mothered_array[:,0]), label = 'Mothered simulators')
# plt.scatter(observation_times, abs(diffs[:,0]), label = 'Difference')
# plt.grid()
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Time since J2000 [seconds]')
# plt.ylabel('Observation [m]')
# plt.title('Observations - X')
# plt.figure()
# plt.scatter(observation_times, abs(rogue_array[:,1]), label = 'Rogue simulators')
# plt.scatter(observation_times, abs(mothered_array[:,1]), label = 'Mothered simulators')
# plt.scatter(observation_times, abs(diffs[:,1]), label = 'Difference')
# plt.grid()
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Time since J2000 [seconds]')
# plt.ylabel('Observation [m]')
# plt.title('Observations - Y')
# plt.figure()
# plt.scatter(observation_times, abs(rogue_array[:,2]), label = 'Rogue simulators')
# plt.scatter(observation_times, abs(mothered_array[:,2]), label = 'Mothered simulators')
# plt.scatter(observation_times, abs(diffs[:,2]), label = 'Difference')
# plt.grid()
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Time since J2000 [seconds]')
# plt.ylabel('Observation [m]')
# plt.title('Observations - Z')
#
# libration = read_vector_history_from_file(cwd + 'ephemeris/translation-a.eph')
# no_libration = read_vector_history_from_file(cwd + 'ephemeris/translation-s.eph')
# epochs = np.array(list(libration.keys())) / 86400.0
# libration = dict2array(libration)[:,[1,2,3]]
# no_libration = dict2array(no_libration)[:,[1,2,3]]
# diffs = libration - no_libration
#
# plt.figure()
# plt.plot(epochs, norm_rows(diffs), marker = '.')
# plt.grid()
# plt.yscale('log')
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$|\Delta\vec r|$ [m]')
# plt.title('Position difference between the librational and non-librational ephemerides')

# current_ephemeris = dict2array(read_vector_history_from_file(cwd + 'ephemeris/translation-a.eph'))
# new_states = dict2array(read_vector_history_from_file(cwd + 'simulation-results/model-a1/2023-09-13 16:57:24.579157/state-history.dat'))
# epochs = current_ephemeris[:,0] / 86400.0
# diffs = new_states[:,1:4] - current_ephemeris[:,1:4]
#
# plt.figure()
# plt.plot(epochs, norm_rows(diffs))
# plt.grid()
# plt.yscale('log')
# plt.xlabel('Time since J2000')
# plt.ylabel(r'$\Delta\vec r$ [m]')
# plt.title('Position differences')

########################################################################################################################
#                                                                                                                      #
#                                               LIBRATION AMPLITUDES                                                   #
#                                                                                                                      #
########################################################################################################################

# bodies = get_solar_system('U')
# dependents = read_vector_history_from_file('ephemeris/associated-dependents/c.dat')
# eccentricity = compute_eccentricity_from_dependent_variables(dependents)
# mean_motion = compute_mean_motion_from_dependent_variables(dependents, bodies.get('Mars').gravitational_parameter)
# dependents = dict2array(dependents)
# longitude_spectrum = fourier_transform(dependents[:,[0,6]])
#
# B1 = find_max_in_range(longitude_spectrum, [0.98*mean_motion, 1.02*mean_motion]) / eccentricity
# B2 = find_max_in_range(longitude_spectrum, [2.0*0.98*mean_motion, 2.0*1.02*mean_motion]) / eccentricity / eccentricity
#
# plt.figure()
# plt.axvline(mean_motion * 86400.0, ls = 'dashed', c = 'r', linewidth = 3, label = 'Mean motion')
# plt.axvline(2.0 * mean_motion * 86400.0, ls = 'dashed', c = 'r', linewidth = 3, label = 'Mean motion')
# plt.loglog(longitude_spectrum[:,0] * 86400.0, longitude_spectrum[:,1] / eccentricity, marker = '.')
# plt.grid()
# plt.xlabel(r'$\omega$ [rad/day]')
#
# I = bodies.get('Phobos').inertia_tensor
# sigma = (I[1,1] - I[0,0]) / I[2,2]
# B1_anal = 2.0 / ( 1.0 - 3.0*sigma )
# B2_anal = 5.0 / 4.0 + 3.0*sigma / 2.0 * 1.0 / ( 4 - 3.0*sigma ) * (5.0/2.0 + 3.0*B1_anal)
#
# dB1 = B1_anal - B1
# dB2 = B2_anal - B2

# coupled = read_vector_history_from_file('ephemeris/true/translation-c.eph')
# wrong_lib = read_vector_history_from_file('simulation-results/model-u/wrong-libration/state-history.dat')
# right_lib = read_vector_history_from_file('simulation-results/model-u/right-libration/state-history.dat')
# uncoupled = read_vector_history_from_file('ephemeris/true/translation-u.eph')
#
# diffs = dict2array(compare_results(coupled, uncoupled, list(coupled.keys())))
# diffs_right = dict2array(compare_results(coupled, right_lib, list(coupled.keys())))
# diffs_wrong = dict2array(compare_results(coupled, wrong_lib, list(coupled.keys())))
# diffs_right_wrong = dict2array(compare_results(right_lib, wrong_lib, list(coupled.keys())))
#
# epochs = diffs[:,0] / 86400.0
# first_month = epochs <= 999999999.0
# plt.figure()
# plt.semilogy(epochs[first_month], norm_rows(diffs[first_month,1:4]), label = 'Wrong amplitude')
# plt.semilogy(epochs[first_month], norm_rows(diffs_wrong[first_month,1:4]), label = 'Wrong amplitude')
# plt.semilogy(epochs[first_month], norm_rows(diffs_right[first_month,1:4]), label = 'Right amplitude')
# plt.semilogy(epochs, norm_rows(diffs_right_wrong[:,1:4]), label = 'Diffs')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.title('Coupling effects')
#
# prueba = dict2array(compare_results(right_lib, uncoupled, list(uncoupled.keys())))
# plt.figure()
# plt.plot(epochs, norm_rows(prueba[:,1:4]))
# plt.grid()

bodies = get_solar_system('U')
coupled_dependents = read_vector_history_from_file('ephemeris/associated-dependents/c.dat')
eccentricity = compute_eccentricity_from_dependent_variables(coupled_dependents)
mean_motion = compute_mean_motion_from_dependent_variables(coupled_dependents, bodies.get('Mars').gravitational_parameter)
B = compute_scaled_libration_amplitude_from_dependent_variables(coupled_dependents)
coupled_dependents = dict2array(coupled_dependents)
libration_fourier = fourier_transform(coupled_dependents[:,[0,6]])
B2 = find_max_in_range(libration_fourier, np.array([1.98, 2.02])*mean_motion) / eccentricity / eccentricity
I = bodies.get('Phobos').inertia_tensor
sigma = (I[1,1] - I[0,0]) / I[2,2]
B1_anal = 2.0 / (1.0 - 3.0*sigma)
B2_anal = (1.0/(4.0-3.0*sigma)) * (5.0+4.5*sigma*B1_anal)

print('PROGRAM COMPLETED SUCCESSFULLY')

