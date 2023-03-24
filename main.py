import numpy as np
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

unperturbed_history = read_vector_history_from_file('everything-works-results/model-a1/unperturbed-states.txt')
epochs_list = list(unperturbed_history.keys())
epochs_array = np.array(epochs_list)

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

baseline_history = read_vector_history_from_file('everything-works-results/model-a1/perturbed-baseline.txt')
baseline_moon = read_vector_history_from_file('everything-works-results/model-a1/baseline-plus-moon.txt')
baseline_galileo = read_vector_history_from_file('everything-works-results/model-a1/baseline-plus-all-galilean-moons.txt')
baseline_saturn = read_vector_history_from_file('everything-works-results/model-a1/baseline-plus-saturn.txt')

# baseline_differences = result2array(norm_history(extract_elements_from_history(compare_results(unperturbed_history, baseline_history, epochs_list), [0, 1, 2])))
moon_over_baseline = result2array(norm_history(extract_elements_from_history(compare_results(baseline_history, baseline_moon, epochs_list), [0, 1, 2])))
galileo_over_baseline = result2array(norm_history(extract_elements_from_history(compare_results(baseline_history, baseline_galileo, epochs_list), [0, 1, 2])))
saturn_over_baseline = result2array(norm_history(extract_elements_from_history(compare_results(baseline_history, baseline_saturn, epochs_list), [0, 1, 2])))

# plt.figure()
# plt.semilogy(epochs_array / 86400.0, baseline_differences[:,1])
# plt.xlabel('Time [days since J2000]')
# plt.ylabel(r'$\Delta r$ [m]')
# plt.title('Discrepancies between baseline and unperturbed dynamics')
# plt.grid()

plt.figure()
plt.semilogy(epochs_array / 86400.0, moon_over_baseline[:,1], label = 'Moon')
plt.semilogy(epochs_array / 86400.0, galileo_over_baseline[:,1], label = 'Galilean moons')
plt.semilogy(epochs_array / 86400.0, saturn_over_baseline[:,1], label = 'Saturn')
plt.yticks([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
plt.xlabel('Time [days since J2000]')
plt.ylabel(r'$\Delta r$ [m]')
plt.title('Effect of different bodies on Phobos\' baseline over 900 days')
plt.legend()
plt.grid()