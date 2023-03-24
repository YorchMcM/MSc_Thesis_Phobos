import numpy as np
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *
from numpy import pi as PI
TWOPI = 2*PI

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')

mars_mu = 42828375815756.1
deimos_period = TWOPI*np.sqrt(23463200**3 / mars_mu)
mars_period = 686.98 * 86400.0
earth_period = 365.25 * 86400.0
mars_earth_synodic_period = 1/abs((1/mars_period)-(1/earth_period))

# unperturbed_history = read_vector_history_from_file('everything-works-results/model-a1/unperturbed-states.txt')
# epochs_list = list(unperturbed_history.keys())
# epochs_array = np.array(epochs_list)

# sun_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-sun.txt')
# earth_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-earth.txt')
# deimos_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-deimos.txt')
# jupiter_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-jupiter.txt')
# saturn_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-saturn.txt')
#
# sun_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, sun_history, epochs_list), [0, 1, 2])))
# earth_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, earth_history, epochs_list), [0, 1, 2])))
# deimos_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, deimos_history, epochs_list), [0, 1, 2])))
# jupiter_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, jupiter_history, epochs_list), [0, 1, 2])))
# saturn_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, saturn_history, epochs_list), [0, 1, 2])))
#
# plt.figure()
# plt.semilogy(epochs_array / 86400.0, sun_differences[:,1], label = 'Sun')
# plt.semilogy(epochs_array / 86400.0, earth_differences[:,1], label = 'Earth')
# plt.semilogy(epochs_array / 86400.0, deimos_differences[:,1], label = 'Deimos')
# plt.semilogy(epochs_array / 86400.0, jupiter_differences[:,1], label = 'Jupiter')
# plt.semilogy(epochs_array / 86400.0, saturn_differences[:,1], label = 'Saturn')
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.title('Effect of different bodies on Phobos\' position over 900 days')
# plt.legend()
# plt.grid()

# baseline_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-baseline.txt')
# baseline_moon = read_vector_history_from_file('everything-works-results/model-a1/baseline-plus-moon.txt')
# baseline_galileo = read_vector_history_from_file('everything-works-results/model-a1/baseline-plus-all-galilean-moons.txt')
# baseline_saturn = read_vector_history_from_file('everything-works-results/model-a1/baseline-plus-saturn.txt')

# baseline_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, baseline_history, epochs_list), [0, 1, 2])))
# moon_over_baseline = result2array(norm_history(extract_elements_from_history(compare_results(baseline_history, baseline_moon, epochs_list), [0, 1, 2])))
# galileo_over_baseline = result2array(norm_history(extract_elements_from_history(compare_results(baseline_history, baseline_galileo, epochs_list), [0, 1, 2])))
# saturn_over_baseline = result2array(norm_history(extract_elements_from_history(compare_results(baseline_history, baseline_saturn, epochs_list), [0, 1, 2])))

# plt.figure()
# plt.semilogy(epochs_array / 86400.0, baseline_differences[:,1])
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.title('Discrepancies between baseline and unperturbed dynamics')
# plt.grid()

# plt.figure()
# plt.semilogy(epochs_array / 86400.0, moon_over_baseline[:,1], label = 'Moon')
# plt.semilogy(epochs_array / 86400.0, galileo_over_baseline[:,1], label = 'Galilean moons')
# plt.semilogy(epochs_array / 86400.0, saturn_over_baseline[:,1], label = 'Saturn')
# plt.yticks([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.title('Effect of different bodies on Phobos\' baseline over 900 days')
# plt.legend()
# plt.grid()

state_history = read_vector_history_from_file('phobos-ephemerides-3500.txt')
dependents = read_vector_history_from_file('a1-dependent-variables-3500.txt')

checks = [0, 0, 0, 1, 0]
epochs_array = np.array(list(state_history.keys()))
keplerian_history = extract_elements_from_history(dependents, list(range(6,12)))
average_mean_motion, orbits = average_mean_motion_over_integer_number_of_orbits(keplerian_history, mars_mu)
print('Average mean motion over', orbits, 'orbits:', average_mean_motion, 'rad/s =', average_mean_motion*86400.0, 'rad/day')
phobos_period = TWOPI / average_mean_motion
phobos_deimos_synodic_period = 1/abs((1/phobos_period)-(1/deimos_period))

phobos_deimos_frequency = TWOPI / phobos_deimos_synodic_period
mars_frequency = TWOPI / mars_period
mars_earth_frequency = TWOPI / mars_earth_synodic_period

# Trajectory
if checks[0]:
    trajectory_3d(state_history, ['Phobos'], 'Mars')

# Orbit does not blow up.
if checks[1]:
    plot_kepler_elements(keplerian_history)

# Orbit is equatorial
if checks[2]:
    sub_phobian_point = result2array(extract_elements_from_history(dependents, [13, 14]))
    sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:,1:], -PI, PI, include = 'upper')

    plt.figure()
    plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:,1] * 360.0 / TWOPI)
    plt.grid()
    plt.title('Sub-phobian point')
    plt.xlabel('LON [º]')
    plt.ylabel('LAT [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.title('Sub-phobian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

# Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
if checks[3]:
    sub_martian_point = result2array(extract_elements_from_history(dependents, [4, 5]))
    sub_martian_point[:,1:] = bring_inside_bounds(sub_martian_point[:,1:], -PI, PI, include = 'upper')
    libration_history = extract_elements_from_history(dependents, 5)
    libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)
    # phobos_mean_rotational_rate = 0.00022785759213999574  # In rad/s
    phobos_mean_rotational_rate = 0.000227995  # In rad/s

    # plt.figure()
    # plt.scatter(sub_martian_point[:,2] * 360.0 / TWOPI, sub_martian_point[:,1] * 360.0 / TWOPI)
    # plt.grid()
    # plt.title('Sub-martian point')
    # plt.xlabel('LON [º]')
    # plt.ylabel('LAT [º]')
    #
    # plt.figure()
    # plt.plot(epochs_array / 86400.0, sub_martian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    # plt.plot(epochs_array / 86400.0, sub_martian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    # plt.legend()
    # plt.grid()
    # plt.title('Sub-martian point')
    # plt.xlabel('Time [days since J2000]')
    # plt.ylabel('Coordinate [º]')

    plt.figure()
    plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker = '.')
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobos\' mean motion')
    plt.axline((2*phobos_mean_rotational_rate * 86400.0, 0), (2*phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r')
    plt.axline((3*phobos_mean_rotational_rate * 86400.0, 0), (3*phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r')
    # plt.axline((phobos_deimos_frequency * 86400.0, 0), (phobos_deimos_frequency * 86400.0, 1), ls='dashed', c='c', label='Phobos-Deimos')
    # plt.axline((mars_frequency * 86400.0, 0), (mars_frequency * 86400.0, 1), ls='dashed', c='g', label='Mars\' mean motion')
    # plt.axline((mars_earth_frequency * 86400.0, 0), (mars_earth_frequency * 86400.0, 1), ls='dashed', c='k', label='Mars-Earth')
    plt.title(r'Libration frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.grid()
    # plt.xlim([0, 21])
    plt.legend()

# Accelerations exerted by all third bodies. This will be used to assess whether the bodies are needed or not.
if checks[4]:
    third_body_accelerations = result2array(extract_elements_from_history(dependents, list(range(15,20))))
    third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
    plt.figure()
    for idx, body in enumerate(third_bodies):
        plt.semilogy(epochs_array / 86400.0, third_body_accelerations[:,idx+1], label = body)
    plt.title('Third body accelerations')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel(r'Acceleration [m/s²]')
    plt.legend()
    plt.grid()