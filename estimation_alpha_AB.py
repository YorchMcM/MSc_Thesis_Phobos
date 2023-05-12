import numpy as np
from time import time
from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import environment_setup, estimation_setup, estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter
from tudatpy.io import save2txt
from tudatpy.kernel.interface import spice

from cycler import cycler
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fman

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')
color_list = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = cycler('color', color_list)
plt.rcParams['lines.markersize'] = 6.0

spice.load_standard_kernels()

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
trajectory_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
imposed_trajectory = extract_elements_from_history(read_vector_history_from_file(trajectory_file), [0, 1, 2, 3, 4, 5])
phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = 0.0 # np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source, scaled_amplitude)

# FIRST, WE PROPAGATE THE DYNAMICS AND WE SET PHOBOS' EPHEMERIS TO THE INTEGRATED RESULTS.
# WE ACTUALLY DON'T NEED THIS FOR THE POSITION OBSERVATIONS.

# PROPAGATOR
simulation_time = 3.0 * constants.JULIAN_YEAR
initial_estimation_epoch = 1.0 * constants.JULIAN_YEAR
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, initial_epoch = initial_estimation_epoch)

# PARAMETERS TO ESTIMATE
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Libration amplitude missing ???
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# LINK SET UP
# link_ends = { observation.LinkEndType.transmitter : observation.body_origin_link_end_id('Mars'),
#               observation.LinkEndType.receiver : observation.body_origin_link_end_id('Phobos') }
link_ends = { observation.observed_body : observation.body_origin_link_end_id('Phobos') }
link = observation.link_definition(link_ends)

# observation_simulators = estimation_setup.create_observation_simulators(observation_settings, bodies)  # What is this going to be used for?

# NOW, WE CREATE THE OBSERVATIONS

t0 = 1.5 * constants.JULIAN_YEAR
tf = 3.5 * constants.JULIAN_YEAR
dt = 20.0 * 60.0  # 20min
N = int((tf - t0) / dt) + 1
observation_times = np.linspace(t0, tf, N)

observable_type = estimation_setup.observation.ObservableType.position_observable_type
observation_simulation_settings = observation.tabulated_simulation_settings(observable_type,
                                                                            link,
                                                                            observation_times,
                                                                            reference_link_end_type=estimation_setup.observation.observed_body)

observation_model_settings = observation.cartesian_position(link)
observation_simulators = estimation_setup.create_observation_simulators([observation_model_settings], bodies)  # This is already a list!

observation_collection = estimation.simulate_observations([observation_simulation_settings],
                                                          observation_simulators,
                                                          bodies)

maximum_number_of_iterations = 10
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = maximum_number_of_iterations)
estimation_input = estimation.EstimationInput(observation_collection, convergence_checker = convergence_checker)

# AND NOW WE CREATE THE ESTIMATOR OBJECT, WE PROPAGATE THE VARIATIONAL EQUATIONS AND WE ESTIMATE
print('Going into the depths of tudat...')
tic = time()
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    [observation_model_settings],
    propagator_settings)
tac = time()
print('We\'re back! Variational equations propagated. Time taken:', (tac - tic) / 60.0, 'min')
print('Performing estimation...')
tic = time()
estimation_output = estimator.perform_estimation(estimation_input)
tac = time()
print('Estimation completed. Time taken:', (tac - tic) / 60.0, 'min')

# SAVE RESULTS
save_dir = getcwd() + '/estimation-ab/alpha/'
number_of_iterations = estimation_output.residual_history.shape[1]
residual_history, parameter_evolution, residual_rms_evolution = extract_estimation_output(estimation_output, list(observation_times), 'position')
# El residual_rms_evolution sigue saliendo mal, y todo es constante !

true_initial_state = get_true_initial_state(bodies, initial_estimation_epoch)

save2txt(residual_history, save_dir + 'residual-history.txt')
save2txt(parameter_evolution, save_dir + 'parameter-evolution.txt')
save2txt(residual_rms_evolution, save_dir + 'rms-evolution.txt')

residual_array = result2array(residual_history)
plt.figure()
for k in range(number_of_iterations):
    plt.plot((residual_array[:,0] - initial_estimation_epoch) / 86400.0, residual_array[:,3*k+1], label = 'x (Iter ' + str(k+1) + ')')
    # plt.plot((residual_array[:,0] - initial_estimation_epoch) / 86400.0, residual_array[:,3*k+1], c = color_list[k], label = 'x (Iter ' + str(k+1) + ')')
    # plt.plot((residual_array[:,0] - initial_estimation_epoch) / 86400.0, residual_array[:,3*k+2], c=color_list[k], label='y (Iter ' + str(k+1) + ')', ls = '--')
    # plt.plot((residual_array[:,0] - initial_estimation_epoch) / 86400.0, residual_array[:,3*k+3], c=color_list[k], label='z (Iter ' + str(k+1) + ')', ls = '-.')
plt.grid()
plt.xlabel('Time since estimation start [days]')
plt.ylabel('Position residuals [m]')
plt.legend()
plt.title('Residual history')

residual_rms_evolution = get_position_rms_evolution(residual_history)
rms_array = result2array(residual_rms_evolution)
plt.figure()
plt.plot(rms_array[:,0], rms_array[:,1], label = 'x', marker = '.')
plt.plot(rms_array[:,0], rms_array[:,2], label = 'y', marker = '.')
plt.plot(rms_array[:,0], rms_array[:,3], label = 'z', marker = '.')
plt.grid()
plt.xlabel('Iteration number')
plt.ylabel('Residual rms [m]')
plt.legend()
plt.title('Residual root mean square')