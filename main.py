import os
import numpy as np
from Auxiliaries import *
from numpy import pi as PI
import myconstants
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fman
TWOPI = 2*PI

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')

average_mean_motion = myconstants.average_mean_motion
normal_mode = myconstants.normal_mode
phobos_mean_rotational_rate = myconstants.phobos_mean_rotational_rate

dissipation_times = [4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0]

read_dir = os.getcwd() + '/everything-works-results/model-a2/'
freq_undamped, amp_undamped = get_fourier_elements_from_history(extract_elements_from_history(
    read_vector_history_from_file(read_dir + 'dependents-undamped.txt'), 5))

freqs, amps = np.zeros([len(freq_undamped), 12]), np.zeros([len(freq_undamped), 12])
for idx, time in enumerate(dissipation_times):
    print(idx, time)
    current_file = read_dir + 'dependents-d' + str(int(time))
    if idx < len(dissipation_times): current_file = current_file + '-full'
    current_file = current_file + '.txt'
    freqs[:,idx], amps[:,idx] = get_fourier_elements_from_history(extract_elements_from_history(
        read_vector_history_from_file(current_file), 5))

fig = plt.figure()
plt.loglog(freq_undamped * 86400.0, amp_undamped * 360 / TWOPI, marker='.', label = 'Undamped')
for idx, time in enumerate(dissipation_times):
    plt.loglog(freqs[:,idx] * 86400.0, amps[:,idx] * 360 / TWOPI, marker='.', label = str(int(time)) + 'h')
plt.axline((0, 0.0014), (1, 0.0014), ls='dashed', c='k', label='Rambaux\'s threshold')
plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed', c='r',
           label='Phobos\' mean motion (and integer multiples)')
plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='b', label='Longitudinal normal mode')
plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
plt.title(r'Libration frequency content for different damping times')
plt.xlabel(r'$\omega$ [rad/day]')
plt.ylabel(r'$A [º]$')
plt.grid()
fig.legend()