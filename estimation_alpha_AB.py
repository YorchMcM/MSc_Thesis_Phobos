'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
'''

import os
from datetime import datetime
from shutil import copyfile

from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import estimation, estimation_setup, Estimator
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

settings = EstimationSettings(os.getcwd() + '/estimation-settings.inp')

########################################################################################################################
# SETTINGS COLLECTION
#
# Estimation
estimation_model = settings.estimation_settings['estimation model']
initial_estimation_epoch = settings.estimation_settings['initial estimation epoch']
duration_of_estimated_arc = settings.estimation_settings['duration of estimated arc']
estimated_parameters = settings.estimation_settings['estimated parameters']

# Observations
observation_model = settings.observation_settings['observation model']
observation_types = settings.observation_settings['observation type']
epoch_of_first_observation = settings.observation_settings['epoch of first observation']
epoch_of_last_observation = settings.observation_settings['epoch of last observation']
observation_frequency = settings.observation_settings['observation frequency']

# Least squares convergence
maximum_number_of_iterations = settings.ls_convergence_settings['maximum number of iterations']
minimum_residual_change = settings.ls_convergence_settings['minimum residual change']
minimum_residual = settings.ls_convergence_settings['minimum residual']
number_of_iterations_without_improvement = settings.ls_convergence_settings['number of iterations without improvement']

# Test functionalities
test_mode = settings.test_functionalities['test mode']
initial_position_perturbation = settings.test_functionalities['initial position perturbation']
initial_velocity_perturbation = settings.test_functionalities['initial velocity perturbation']
initial_orientation_perturbation = settings.test_functionalities['initial orientation perturbation']
initial_angular_velocity_perturbation = settings.test_functionalities['initial angular velocity perturbation']
apply_perturbation_in_rsw = settings.test_functionalities['apply perturbation in rsw']

# Execution
save = settings.execution_settings['save results']
save_state_history_per_iteration = settings.execution_settings['save state history per iteration']
post_process_in_this_file = settings.execution_settings['post process in present run']

########################################################################################################################

if settings.execution_settings['save results']:
    estimation_type = settings.get_estimation_type()
    if estimation_type is None:
        save_dir = os.getcwd() + '/estimation-results/' + str(datetime.now()) + '/'
    else:
        save_dir = os.getcwd() + '/estimation-results/' + estimation_type + '/' + str(datetime.now()) + '/'
    os.makedirs(save_dir)
    copyfile(settings.source_file, save_dir + 'settings.log')


# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
translational_ephemeris_file, rotational_ephemeris_file = retrieve_ephemeris_files(observation_model)
bodies = get_solar_system(estimation_model, translational_ephemeris_file, rotational_ephemeris_file)

# ESTIMATION DYNAMICS

if estimation_model != 'A2':
    true_translational_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch)
if estimation_model in ['A2', 'B', 'C']:
    true_rotational_state = bodies.get('Phobos').rotation_model.interpolator.interpolate(initial_estimation_epoch)

if test_mode:
    if estimation_model != 'A2':
        if apply_perturbation_in_rsw:
            RSW_R_I = inertial_to_rsw_rotation_matrix(true_translational_state)
            R = np.concatenate((np.concatenate((RSW_R_I, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), RSW_R_I), 1)), 0)
        else: R = np.eye(6)
        translational_state = true_translational_state + R.T @ np.concatenate((initial_position_perturbation, initial_velocity_perturbation))
    if estimation_model in ['A2', 'B', 'C']:
        rotational_state = true_rotational_state + np.concatenate((initial_orientation_perturbation, initial_angular_velocity_perturbation))
else: translational_state = true_translational_state

if estimation_model in ['S', 'A1']:
    initial_state = translational_state
    true_initial_state = true_translational_state
elif estimation_model == ['A2']:
    initial_state = rotational_state
    true_initial_state = true_rotational_state
else:
    initial_state = np.concatenate((translational_state, rotational_state))
    true_initial_state = np.concatenate((true_translational_state, true_rotational_state))

initial_state = settings.get_estimation_initial_state()
propagator_settings = get_propagator_settings(estimation_model, bodies, initial_estimation_epoch, initial_state, duration_of_estimated_arc)


# PARAMETERS TO ESTIMATE
parameters_str = ''
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
parameters_str = parameters_str + '\t- Initial state\n'
if 'A' in estimated_parameters:
    # AQUÍ ME FALTA EXPONER LA LIBRATION AMPLITUDE COMO ESTIMATABLE PARAMETER (CREO)
    pass
if 'C20' in estimated_parameters and 'C22' in estimated_parameters:
    parameter_settings = parameter_settings + [estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2,0),(2,2)])]
    parameters_str = parameters_str + '\t- C20\n'
    parameters_str = parameters_str + '\t- C22\n'
elif 'C20' in estimated_parameters:
    parameter_settings = parameter_settings + [estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2,0)])]
    parameters_str = parameters_str + '\t- C20\n'
elif 'C22' in estimated_parameters:
    parameter_settings = parameter_settings + [estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2,2)])]
    parameters_str = parameters_str + '\t- C22\n'

parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)



# OBSERVATION SIMULATION
#Link set up
link_ends = { observation.observed_body : observation.body_origin_link_end_id('Phobos') }
link = observation.LinkDefinition(link_ends)
# Observation creation
observation_model_settings = observation.cartesian_position(link)
N = int((epoch_of_last_observation - epoch_of_first_observation) / observation_frequency) + 1
observation_times = np.linspace(epoch_of_first_observation, epoch_of_last_observation, N)

observable_type = observation.position_observable_type
observation_simulation_settings = observation.tabulated_simulation_settings(observable_type,
                                                                            link,
                                                                            observation_times,
                                                                            reference_link_end_type=observation.observed_body)

observation_simulators = estimation_setup.create_observation_simulators([observation_model_settings], bodies)  # This is already a list!

observation_collection = estimation.simulate_observations([observation_simulation_settings],
                                                          observation_simulators,
                                                          bodies)

convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = maximum_number_of_iterations,
                                                                minimum_residual_change = minimum_residual_change,
                                                                minimum_residual = minimum_residual,
                                                                number_of_iterations_without_improvement = number_of_iterations_without_improvement)
estimation_input = estimation.EstimationInput(observation_collection, convergence_checker = convergence_checker)
estimation_input.define_estimation_settings(save_state_history_per_iteration)

# We will first save the log, before starting the possibly very long estimation process.
if save:
    perturbation = np.concatenate((initial_position_perturbation, initial_velocity_perturbation))
    log = '\nESTIMATION\n· Estimation model: ' + estimation_model + '\n· Initial epoch: ' + \
          str(initial_estimation_epoch/86400.0) + ' days\n· Final epoch: ' + \
          str((initial_estimation_epoch+duration_of_estimated_arc)/86400.0) + ' days\n· True initial state: ' \
          + str(true_initial_state) + '\n· Perturbation: ' + str(perturbation) + '\n· Estimated parameters:\n' + \
          parameters_str + '\nOBSERVATIONS\n· Observation model: ' + observation_model + '\n· Ephemeris file: ' + \
          str(translational_ephemeris_file) + '\n· Rotation file: ' + str(rotational_ephemeris_file) + \
          '\n· Epoch of first observation: ' + str(epoch_of_first_observation / 86400.0) + \
          ' days\n· Epoch of last observation: ' + str(epoch_of_last_observation / 86400.0) + ' days\n· Frequency: ' + \
          str(observation_frequency / 60.0) + ' min\n\nCONVERGENCE\n' + '· Maximum number of iterations: ' + \
          str(maximum_number_of_iterations) + '\n· Minimum residual change: ' + str(minimum_residual_change) + \
          '\n· Minimum residual: ' + str(minimum_residual) + '\n· Number of iterations without improvement: ' + \
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

if apply_perturbation_in_rsw:
    RSW_R_I = inertial_to_rsw_rotation_matrix(true_translational_state)
    R = np.concatenate((np.concatenate((RSW_R_I, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), RSW_R_I), 1)), 0)
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
