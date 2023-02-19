import numpy as np
from os import getcwd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from benchmark_generation import generate_benchmark_errors
from Auxiliaries import norm_rows, read_vector_history_from_file

from tudatpy.kernel import constants
# from tudatpy.io import read_vector_history_from_file
from tudatpy.util import result2array

generate_benchmarks = False

for font in fman.findSystemFonts(r'C:\Users\Yorch\OneDrive - Delft University of Technology\Year 2022-2023\MSc_Thesis_Phobos\Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

errors_dir = getcwd() + '/benchmark_errors'
step_sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # These are in MINUTES
simulation_time = 90.0*constants.JULIAN_DAY
# simulation_time = 20.0*constants.JULIAN_YEAR
if generate_benchmarks:
    generate_benchmark_errors(step_sizes, simulation_time)

max_errors = []
simulation_time = simulation_time / constants.JULIAN_DAY
time_unit = 'days'
if simulation_time > 1000.0:
    simulation_time = simulation_time / constants.JULIAN_YEAR_IN_DAYS
    time_unit = 'years'
for step_size in step_sizes:
    filename = errors_dir + '/benchmark_errors_' + str(simulation_time).removesuffix('.0') + '_' + str(step_size).removesuffix('.0') + '.txt'
    errors = result2array(read_vector_history_from_file(filename))
    normed_errors = norm_rows(errors[:,1:4])
    max_errors.append(max(normed_errors))

plt.semilogy(np.array(step_sizes), np.array(max_errors), '.-')
plt.xlabel('Time step size [min]')
plt.ylabel(r'$\Delta r_{\mathrm{max}}$ [m]')
plt.title('Max benchmark errors after ' + str(simulation_time).removesuffix('.0') + ' ' + time_unit)
plt.grid()
plt.show(block = True)