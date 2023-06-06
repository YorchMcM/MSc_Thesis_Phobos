'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
'''

import os
from datetime import datetime

from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import estimation, estimation_setup, Estimator
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

save = True
save_state_history_per_iteration = False
post_process_in_this_file = True

observation_model = 'Synch'
estimation_model = 'Synch'

if save:
    save_dir = os.getcwd() + '/estimation-alpha-test/' + str(datetime.now()) + '/'
    os.makedirs(save_dir)

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
if observation_model == 'Synch':
    trajectory_file = '/home/yorch/thesis/phobos-ephemerides-3500-nolib.txt'
elif observation_model == 'A1':
    trajectory_file = '/home/yorch/thesis/phobos-ephemerides-3500.txt'
elif observation_model == 'B':
    trajectory_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
else:
    raise ValueError('Invalid observation model selected.')

phobos_ephemerides = get_ephemeris_from_file(trajectory_file)
if estimation_model == 'Synch': libration_amplitude = 0.0
else: libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, scaled_amplitude)

# PROPAGATOR
simulation_time = 1.0 * constants.JULIAN_YEAR
initial_estimation_epoch = 1.0 * constants.JULIAN_YEAR
position_perturbation = np.array([500.0, -800.0, 100.0])
velocity_perturbation = np.array([0.02, 0.01, -0.07])
if estimation_model in ['A1', 'Synch']:
    perturbation = np.concatenate((position_perturbation, velocity_perturbation))
    perturbed_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch) + perturbation
    propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, initial_estimation_epoch, initial_state = perturbed_initial_state)
if estimation_model == 'B':
    perturbation = np.zeros(13)
    perturbation[:3] = position_perturbation
    perturbed_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch) + perturbation
    propagator_settings = get_model_b_propagator_settings(bodies, simulation_time, initial_epoch = initial_estimation_epoch,
                                                          initial_state = perturbed_initial_state)

true_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch)
RSW_R_I = inertial_to_rsw_rotation_matrix(true_initial_state)
R = np.concatenate((np.concatenate((RSW_R_I, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), RSW_R_I), 1)), 0)
# PARAMETERS TO ESTIMATE
parameters_str = ''
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
parameters_str = parameters_str + '\t- Initial state\n'
# parameter_settings = parameter_settings + [estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2,0),(2,2)])]
# parameters_str = parameters_str + '\t- C20\n'
# parameters_str = parameters_str + '\t- C22\n'
# Libration amplitude missing ???
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# LINK SET UP
link_ends = { observation.observed_body : observation.body_origin_link_end_id('Phobos') }
link = observation.LinkDefinition(link_ends)

# NOW, WE CREATE THE OBSERVATIONS
observation_model_settings = observation.cartesian_position(link)

t0 = 1.0 * constants.JULIAN_YEAR + 86400.0
tf = 1.0 * constants.JULIAN_YEAR + simulation_time - 86400.0
dt = 20.0 * 60.0  # 20min
N = int((tf - t0) / dt) + 1
observation_times = np.linspace(t0, tf, N)

observable_type = observation.position_observable_type
observation_simulation_settings = observation.tabulated_simulation_settings(observable_type,
                                                                            link,
                                                                            observation_times,
                                                                            reference_link_end_type=observation.observed_body)

observation_simulators = estimation_setup.create_observation_simulators([observation_model_settings], bodies)  # This is already a list!

observation_collection = estimation.simulate_observations([observation_simulation_settings],
                                                          observation_simulators,
                                                          bodies)

maximum_number_of_iterations = 10
minimum_residual_change = 0.0
minimum_residual = 0.0
number_of_iterations_without_improvement = 2
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = maximum_number_of_iterations,
                                                                minimum_residual_change = minimum_residual_change,
                                                                minimum_residual = minimum_residual,
                                                                number_of_iterations_without_improvement = number_of_iterations_without_improvement)
estimation_input = estimation.EstimationInput(observation_collection, convergence_checker = convergence_checker)
estimation_input.define_estimation_settings(save_state_history_per_iteration)

# We will first save the log, before starting the possibly very long estimation process.
if save:
    log = '\nOBSERVATIONS\n· Observation model: ' + observation_model + '\n· Ephemeris file: ' + trajectory_file + \
          '\n· Epoch of first observation: ' + str(t0 / 86400.0) + ' days\n· Epoch of last observation: ' + str(tf / 86400.0) + \
          ' days\n· Frequency: ' + str(dt / 60.0) + ' min\n\nESTIMATION\n· Estimation model: ' + estimation_model + \
          '\n· Libration amplitude: ' + str(libration_amplitude) + ' degrees\n· Initial epoch: ' + str(initial_estimation_epoch/86400.0) + \
          ' days\n· Final epoch: ' + str((initial_estimation_epoch+simulation_time)/86400.0) + ' days\n· True initial state: ' + str(true_initial_state) + \
          '\n· Perturbation: ' + str(perturbation) + '\n· Estimated parameters:\n' + parameters_str + '\nCONVERGENCE\n' + \
          '· Maximum number of iterations: ' + str(maximum_number_of_iterations) + '\n· Minimum residual change: ' + \
          str(minimum_residual_change) + '\n· Minimum residual: ' + str(minimum_residual) + '\n· Number of iterations without improvement: ' + \
          str(number_of_iterations_without_improvement)

    with open(save_dir + 'log.txt', 'w') as file: file.write(log)

# AND NOW WE CREATE THE ESTIMATOR OBJECT, WE PROPAGATE THE VARIATIONAL EQUATIONS AND WE ESTIMATE
print('Going into the depths of tudat...')
tic = time()
estimator = Estimator(
    bodies,
    parameters_to_estimate,
    [observation_model_settings],
    propagator_settings)
tac = time()
print('We\'re back!')
print('Performing estimation...')
tic = time()
estimation_output = estimator.perform_estimation(estimation_input)
tac = time()
print('Estimation completed. Time taken:', (tac - tic) / 60.0, 'min')

residual_history, parameter_evolution, residual_rms_evolution = extract_estimation_output(estimation_output, list(observation_times), 'position')

rsw_covariance = R @ estimation_output.covariance[:6,:6] @ R.T
rsw_correlation = covariance_to_correlation(rsw_covariance)

if save:

    # SAVE RESULTS

    save2txt(residual_history, save_dir + 'residual-history.dat')
    save2txt(parameter_evolution, save_dir + 'parameter-evolution.dat')
    save2txt(residual_rms_evolution, save_dir + 'rms-evolution.dat')
    save_matrix_to_file(estimation_output.covariance, save_dir + 'inertial_covariance.cov')
    save_matrix_to_file(rsw_covariance, save_dir + 'rsw_covariance.cov')
    save_matrix_to_file(estimation_output.correlations, save_dir + 'inertial_correlations.cor')
    save_matrix_to_file(rsw_correlation, save_dir + 'rsw_correlations.cor')


if post_process_in_this_file:

    residual_history_array = result2array(residual_history)
    parameter_evolution_array = result2array(parameter_evolution)
    rms_array = result2array(residual_rms_evolution)

    number_of_iterations = int(parameter_evolution_array.shape[0] - 1)

    for k in range(number_of_iterations):
        plt.figure()
        plt.plot((residual_history_array[:, 0] - initial_estimation_epoch) / 86400.0,
                 residual_history_array[:, 3 * k + 1] / 1000.0, label='x')
        plt.plot((residual_history_array[:, 0] - initial_estimation_epoch) / 86400.0,
                 residual_history_array[:, 3 * k + 2] / 1000.0, label='y')
        plt.plot((residual_history_array[:, 0] - initial_estimation_epoch) / 86400.0,
                 residual_history_array[:, 3 * k + 3] / 1000.0, label='z')
        plt.grid()
        plt.xlabel('Time since estimation start [days]')
        plt.ylabel('Position residuals [km]')
        plt.legend()
        plt.title('Residual history (iteration ' + str(k + 1) + ')')

    plt.figure()
    plt.plot(rms_array[:, 0], rms_array[:, 1] / 1000.0, label='x', marker='.')
    plt.plot(rms_array[:, 0], rms_array[:, 2] / 1000.0, label='y', marker='.')
    plt.plot(rms_array[:, 0], rms_array[:, 3] / 1000.0, label='z', marker='.')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Residual rms [km]')
    plt.legend()
    plt.title('Residual root mean square')

    plt.figure()
    plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 1] / 1000.0, label=r'$x_o$', marker='.')
    plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 2] / 1000.0, label=r'$y_o$', marker='.')
    plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 3] / 1000.0, label=r'$z_o$', marker='.')
    plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 4], label=r'$v_{x,o}$', marker='.')
    plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 5], label=r'$v_{y,o}$', marker='.')
    plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 6], label=r'$v_{z,o}$', marker='.')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Parameter value [km | m/s]')
    plt.legend()
    plt.title('Parameter history')

    plt.figure()
    plt.semilogy(parameter_evolution_array[:,0], abs(parameter_evolution_array[:,1] - true_initial_state[0]), label=r'$x_o$', marker='.')
    plt.semilogy(parameter_evolution_array[:,0], abs(parameter_evolution_array[:,2] - true_initial_state[1]), label=r'$y_o$', marker='.')
    plt.semilogy(parameter_evolution_array[:,0], abs(parameter_evolution_array[:,3] - true_initial_state[2]), label=r'$z_o$', marker='.')
    plt.semilogy(parameter_evolution_array[:,0], abs(parameter_evolution_array[:,4] - true_initial_state[3])*1000.0, label=r'$v_{x,o}$', marker='.')
    plt.semilogy(parameter_evolution_array[:,0], abs(parameter_evolution_array[:,5] - true_initial_state[4])*1000.0, label=r'$v_{y,o}$', marker='.')
    plt.semilogy(parameter_evolution_array[:,0], abs(parameter_evolution_array[:,6] - true_initial_state[5])*1000.0, label=r'$v_{z,o}$', marker='.')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Parameter difference from truth [m | mm/s]')
    plt.legend()
    plt.title('Parameter history')

    parameter_changes = np.zeros([number_of_iterations, len(parameter_evolution[0])])
    for k in range(number_of_iterations):
        parameter_changes[k,:] = parameter_evolution[k+1] - parameter_evolution[k]

    plt.figure()
    plt.plot(parameter_evolution_array[1:, 0], abs(parameter_changes[:, 0]) / 1000.0, label=r'$x_o$', marker='.')
    plt.plot(parameter_evolution_array[1:, 0], abs(parameter_changes[:, 1]) / 1000.0, label=r'$y_o$', marker='.')
    plt.plot(parameter_evolution_array[1:, 0], abs(parameter_changes[:, 2]) / 1000.0, label=r'$z_o$', marker='.')
    plt.plot(parameter_evolution_array[1:, 0], abs(parameter_changes[:, 3]), label=r'$v_{x,o}$', marker='.')
    plt.plot(parameter_evolution_array[1:, 0], abs(parameter_changes[:, 4]), label=r'$v_{y,o}$', marker='.')
    plt.plot(parameter_evolution_array[1:, 0], abs(parameter_changes[:, 5]), label=r'$v_{z,o}$', marker='.')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Parameter change [km | m/s]')
    plt.legend()
    plt.title('Parameter change between pre- and post-fit')

print('PROGRAM COMPLETED SUCCESSFULLY')
