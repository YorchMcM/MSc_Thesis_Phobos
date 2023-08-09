'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
'''

from shutil import copyfile

from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import estimation, estimation_setup, Estimator
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

eccentricity = 0.015034167790105173

print('Reading input file...')
settings = EstimationSettings(os.getcwd() + '/estimation-settings.inp')

true_parameters = settings.get_true_parameters()

########################################################################################################################
# SETTINGS COLLECTION
#
# Estimation
estimation_model = settings.estimation_settings['estimation model']
initial_estimation_epoch = settings.estimation_settings['initial estimation epoch']
duration_of_estimated_arc = settings.estimation_settings['duration of estimated arc']
estimated_parameters = settings.estimation_settings['estimated parameters']
norm_position_residuals = settings.estimation_settings['norm position residuals']
convert_residuals_to_rsw = settings.estimation_settings['convert residuals to rsw']

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

# Post processing settings
post_process_in_this_file = settings.postprocess_settings['post process in present run']
plot_observations = settings.postprocess_settings['plot observations']
plot_cartesian_residuals = settings.postprocess_settings['plot cartesian residuals']
plot_normed_residuals = settings.postprocess_settings['plot normed residuals']
plot_rsw_residuals = settings.postprocess_settings['plot rsw residuals']

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

########################################################################################################################

if type(estimated_parameters) != list:
    estimated_parameters = [estimated_parameters]

if not test_mode:
    apply_perturbation_in_rsw = False

if save:
    estimation_type = settings.get_estimation_type()
    current_run = str(datetime.now())
    # current_run = 'test-for-observation-plots'
    if estimation_type is None:
        save_dir = os.getcwd() + '/estimation-results/' + current_run + '/'
        logs_dir = os.getcwd() + '/estimation-results/logs/'
    else:
        save_dir = os.getcwd() + '/estimation-results/' + estimation_type + '/' + current_run + '/'
        logs_dir = os.getcwd() + '/estimation-results/' + estimation_type + '/logs/'
    os.makedirs(save_dir, exist_ok = True)
    os.makedirs(logs_dir, exist_ok = True)
    copyfile(settings.source_file, save_dir + 'settings.log')
    copyfile(settings.source_file, logs_dir + current_run + '.log')

print('Setting up estimation...')

# CREATE YOUR UNIVERSE FOR OBSERVATION SIMULATION
translational_ephemeris_file, rotational_ephemeris_file = retrieve_ephemeris_files(observation_model, use_new = True)
bodies = get_solar_system(observation_model, translational_ephemeris_file, rotational_ephemeris_file)

# OBSERVATION SIMULATION
#Link set up
link_ends = { observation.observed_body : observation.body_origin_link_end_id('Phobos') }
link = observation.LinkDefinition(link_ends)
# Observationa simulators creation
observation_model_settings = observation.cartesian_position(link, None)
observation_simulators = estimation_setup.create_observation_simulators([observation_model_settings], bodies)  # This is already a list!
# Observation simulation settings
observation_times = settings.observation_settings['observation times']

observable_type = observation.position_observable_type
observation_simulation_settings = observation.tabulated_simulation_settings(observable_type,
                                                                            link,
                                                                            observation_times,
                                                                            reference_link_end_type=observation.observed_body)
# Observation simulation (per se)
observation_collection = estimation.simulate_observations([observation_simulation_settings],
                                                          observation_simulators,
                                                          bodies)

# CREATE YOUR UNIVERSE FOR ESTIMATION
translational_ephemeris_file, TRASH = retrieve_ephemeris_files(estimation_model)
bodies = get_solar_system(estimation_model, translational_ephemeris_file)

# ESTIMATION DYNAMICS
true_initial_state = settings.get_initial_state('true')  # Here, the observation ephemeris are used to retrieve the state.
initial_state = settings.get_initial_state('estimation')  # Here, the estimation ephemeris are used to retrieve the state. Furthermore, if test mode is on, it is also perturbed.
propagator_settings = get_propagator_settings(estimation_model, bodies, initial_estimation_epoch, initial_state, duration_of_estimated_arc)


# PARAMETERS TO ESTIMATE
parameters_to_estimate, parameters_str = get_parameter_set(estimated_parameters, bodies, propagator_settings, return_only_settings_list = False)
# parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies) + parameter_settings
# parameters_str = '\t- Initial state\n' + parameters_str
# parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
print('\nParameters to be estimated:\n' + parameters_str)


# LEAST SQUARES CONVERGENCE
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = maximum_number_of_iterations,
                                                                minimum_residual_change = minimum_residual_change,
                                                                minimum_residual = minimum_residual,
                                                                number_of_iterations_without_improvement = number_of_iterations_without_improvement)


# ESTIMATION INPUT OBJECT
estimation_input = estimation.EstimationInput(observation_collection, convergence_checker = convergence_checker)
estimation_input.define_estimation_settings(save_state_history_per_iteration = save_state_history_per_iteration)


# LOG
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
print('We\'re back! After', (tac - tic) / 60.0, 'min')
print('Performing estimation...')
tic = time()
estimation_output = estimator.perform_estimation(estimation_input)
tac = time()
print('Estimation completed. Time taken:', (tac - tic) / 60.0, 'min')


# WE EXTRACT THE RELEVANT INFORMATION FROM THE ESTIMATION OUTPUT AND SAVE IT IF REQUIRED
observation_history = get_observation_history(list(observation_times), observation_collection)
residual_histories, parameter_evolution, residual_statistical_indicators_per_iteration = \
    extract_estimation_output(estimation_output,
                              settings)

if apply_perturbation_in_rsw:

    if len(true_initial_state) == 6: true_translational_state = true_initial_state
    elif len(true_initial_state) == 13: true_translational_state = true_initial_state[:6]
    else: true_translational_state = None

    if true_translational_state is not None:

        RSW_R_I = inertial_to_rsw_rotation_matrix(true_translational_state)
        R = np.concatenate((np.concatenate((RSW_R_I, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), RSW_R_I), 1)), 0)
        rsw_covariance = R @ estimation_output.covariance[:6,:6] @ R.T
        rsw_correlation = covariance_to_correlation(rsw_covariance)

if save:

    # SAVE RESULTS

    print('Saving results...')

    save2txt(residual_histories[0], save_dir + 'residual-history-cart.dat')
    if norm_position_residuals:
        save2txt(residual_histories[1], save_dir + 'residual_history-norm.dat')
    if convert_residuals_to_rsw:
        save2txt(residual_histories[-1], save_dir + 'residual_history-rsw.dat')
    save2txt(parameter_evolution, save_dir + 'parameter-evolution.dat')
    save2txt(residual_statistical_indicators_per_iteration[0], save_dir + 'residual-indicators-per-iteration-cart.dat')
    if norm_position_residuals:
        save2txt(residual_statistical_indicators_per_iteration[1], save_dir + 'residual-indicators-per-iteration-norm.dat')
    if convert_residuals_to_rsw:
        save2txt(residual_statistical_indicators_per_iteration[-1], save_dir + 'residual-indicators-per-iteration-rsw.dat')
    save2txt(observation_history, save_dir + 'observation-history.dat')
    save_matrix_to_file(estimation_output.covariance, save_dir + 'covariance-matrix.cov')
    save_matrix_to_file(estimation_output.correlations, save_dir + 'correlation-matrix.cor')

    if save_state_history_per_iteration:
        for idx, simulator in enumerate(estimation_output.simulation_results_per_iteration):
            save2txt(simulator.dynamics_results.state_history, save_dir + 'state-history-iteration-' + str(idx) + '.dat')

    if apply_perturbation_in_rsw:
        if true_translational_state is not None:
            save_matrix_to_file(rsw_covariance, save_dir + 'rsw_covariance.cov')
            save_matrix_to_file(rsw_correlation, save_dir + 'rsw_correlations.cor')


if post_process_in_this_file:

    # POST PROCESS

    print('Generating plots...')

    parameter_evolution_array = result2array(parameter_evolution)
    parameter_evolution_array = settings.convert_libration_amplitude(parameter_evolution_array)
    number_of_iterations = int(parameter_evolution_array.shape[0] - 1)

    R = inertial_to_rsw_rotation_matrix(true_initial_state[:6])
    full_state_rotation_matrix = np.concatenate((np.concatenate((R, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), R), 1)), 0)

    if plot_observations:

        observation_array = result2array(observation_history)

        # VISUALIZATION OF OBSERVATIONS
        plt.figure()
        plt.scatter((observation_array[:,0]-initial_estimation_epoch)/86400.0, observation_array[:,1]/1000.0, label=r'$x$')
        plt.scatter((observation_array[:,0]-initial_estimation_epoch)/86400.0, observation_array[:,2]/1000.0, label=r'$y$')
        plt.scatter((observation_array[:,0]-initial_estimation_epoch)/86400.0, observation_array[:,3]/1000.0, label=r'$z$')
        plt.grid()
        plt.legend()
        plt.title('Observations')
        plt.xlabel('Time since estimation initial epoch [days]')
        plt.ylabel('Observation [km]')

    if plot_cartesian_residuals:

        # CARTESIAN RESIDUAL HISTORIES AND INDICATORS FOR ALL ITERATIONS

        residual_history_array = result2array(residual_histories[0])
        indicators_array = result2array(residual_statistical_indicators_per_iteration[0])
        for k in range(number_of_iterations+1):
            plt.figure()
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0,
                     residual_history_array[:,3*k+1], label='x')
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0,
                     residual_history_array[:,3*k+2], label='y')
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0,
                     residual_history_array[:,3*k+3], label='z')
            plt.grid()
            plt.xlabel('Time since estimation start [days]')
            plt.ylabel(r'$\vec\varepsilon_{e,i}$ [m]')
            plt.legend()
            plt.title('Post-fit residual history (iteration ' + str(k) + ')')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,1], marker = '.', color = 'k', ls = '--', label = r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,2], marker='.', color = colors[0], label = r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,3], marker='x', color = colors[1], label = r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,4], marker='.', color= 'k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'$\mu$($x_e$) [m]')
        plt.legend()
        plt.title('Post fit residual mean - $x$ component')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,5], marker = '.', color = 'k', ls = '--', label = r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,6], marker='.', color = colors[0], label = r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,7], marker='x', color = colors[1], label = r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,8], marker='.', color= 'k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'$\mu$($y_e$) [m]')
        plt.legend()
        plt.title('Post fit residual mean - $y$ component')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,9], marker = '.', color = 'k', ls = '--', label = r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,10], marker='.', color = colors[0], label = r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,11], marker='x', color = colors[1], label = r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,12], marker='.', color= 'k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'$\mu$($z_e$) [m]')
        plt.legend()
        plt.title('Post fit residual mean - $z$ component')

        print('\nCARTESIAN RESIDUALS PFRI')
        print(str(indicators_array[-1, 1:]))

    if plot_normed_residuals and norm_position_residuals:

        # NORMED RESIDUAL HISTORIES AND INDICATORS FOR ALL ITERATIONS

        residual_history_array = result2array(residual_histories[1])
        indicators_array = result2array(residual_statistical_indicators_per_iteration[1])
        for k in range(number_of_iterations+1):
            if k == 0: title = 'Pre-fit residual history'
            else: title = 'Post-fit residual history (iteration ' + str(k) + ')'
            plt.figure()
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0, residual_history_array[:,k+1])
            plt.grid()
            plt.xlabel('Time since estimation start [days]')
            plt.ylabel(r'|$\vec\varepsilon_e$| [m]')
            plt.title(title)

        plt.figure()
        plt.semilogy(indicators_array[:,0], indicators_array[:,1], marker = '.', color = 'k', ls = '--', label = r'Min/Max')
        plt.semilogy(indicators_array[:,0], indicators_array[:,2], marker='.', color = colors[0], label = r'Mean')
        plt.semilogy(indicators_array[:,0], indicators_array[:,3], marker='x', color = colors[1], label = r'RMS')
        plt.semilogy(indicators_array[:,0], indicators_array[:,4], marker='.', color= 'k', ls='--')
        plt.grid()
        plt.legend()
        plt.xlabel('Iteration number')
        plt.ylabel(r'Indicator [m]')
        plt.title('Post fit residual evolution - norm')

        print('\nNORMED RESIDUALS PFRI')
        print(str(indicators_array[-1, 1:]))

    if plot_rsw_residuals and convert_residuals_to_rsw:

        # RSW RESIDUAL HISTORIES AND INDICATORS FOR ALL ITERATIONS

        residual_history_array = result2array(residual_histories[-1])
        indicators_array = result2array(residual_statistical_indicators_per_iteration[-1])
        for k in range(number_of_iterations + 1):
            plt.figure()
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0,
                     residual_history_array[:,3*k+1], label='R')
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0,
                     residual_history_array[:,3*k+2], label='S')
            plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0,
                     residual_history_array[:,3*k+3], label='W')
            plt.grid()
            plt.xlabel('Time since estimation start [days]')
            plt.ylabel(r'$\vec\varepsilon_{e,i}$ [m]')
            plt.legend()
            plt.title('Post-fit residual history (iteration ' + str(k) + ')')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,1], marker='.', color='k', ls='--', label=r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,2], marker='.', color=colors[0], label=r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,3], marker='x', color=colors[1], label=r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,4], marker='.', color='k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'$\mu$($R_e$) [m]')
        plt.legend()
        plt.title('Post fit residual mean - $R$ component')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,5], marker='.', color='k', ls='--', label=r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,6], marker='.', color=colors[0], label=r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,7], marker='x', color=colors[1], label=r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,8], marker='.', color='k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'$\mu$($S_e$) [m]')
        plt.legend()
        plt.title('Post fit residual mean - $S$ component')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,9], marker='.', color='k', ls='--', label=r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,10], marker='.', color=colors[0], label=r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,11], marker='x', color=colors[1], label=r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,12], marker='.', color='k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'$\mu$($W_e$) [m]')
        plt.legend()
        plt.title('Post fit residual mean - $W$ component')

        print('\nRSW RESIDUALS PFRI')
        print(str(indicators_array[-1, 1:]))

    # # PARAMETER HISTORY (IN PLAIN VALUES)
    # plt.figure()
    # plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 1] / 1000.0, label=r'$x_o$', marker='.')
    # plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 2] / 1000.0, label=r'$y_o$', marker='.')
    # plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 3] / 1000.0, label=r'$z_o$', marker='.')
    # plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 4], label=r'$v_{x,o}$', marker='.')
    # plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 5], label=r'$v_{y,o}$', marker='.')
    # plt.plot(parameter_evolution_array[:, 0], parameter_evolution_array[:, 6], label=r'$v_{z,o}$', marker='.')
    # if estimation_type in ['bravo', 'charlie']:
    #     plt.plot(parameter_evolution_array[:,0], np.degrees((parameter_evolution_array[:,7] - 2.0)*eccentricity), label = r'$A$ [º]', marker = '.')
    # if estimation_type in []
    # plt.grid()
    # plt.xlabel('Iteration number')
    # plt.ylabel('Parameter value [km | m/s]')
    # plt.legend()
    # plt.title('Parameter history')

    # PARAMETER HISTORY (AS DIFFERENCES WRT TRUTH)
    true_parameters = settings.get_true_parameters()
    plot_legends = settings.get_plot_legends()
    plt.figure()
    for idx in range(len(true_parameters)):
        plt.semilogy(parameter_evolution_array[:,0], 100.0*abs(parameter_evolution_array[:,idx+1] / true_parameters[idx] - 1.0), label = plot_legends[idx], marker = '.')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel(r'$\Delta p$ [% of truth]')
    plt.legend()
    plt.title('Parameter difference from truth')

    # INITIAL STATE DIFFERENCE HISTORY FROM TRUTH IN RSW (AT TRUE INITIAL STATE)
    initial_state_errors = parameter_evolution_array[:,1:7].copy() - true_initial_state[:6]
    for idx in range(len(initial_state_errors)):
        initial_state_errors[idx,:] = full_state_rotation_matrix @ initial_state_errors[idx,:]
        initial_state_errors[idx,:3] = 100.0 * initial_state_errors[idx,:3] / norm(true_initial_state[:3])
        initial_state_errors[idx,3:6] = 100.0 * initial_state_errors[idx,3:6] / norm(true_initial_state[3:6])

    current_legend_list = [r'$R_o$', r'$S_o$', r'$W_o$', r'$v_{r,o}$', r'$v_{s,o}$', r'$v_{w,o}$']
    plt.figure()
    for idx in range(6):
        plt.semilogy(parameter_evolution_array[:,0], abs(initial_state_errors[:,idx]), label = current_legend_list[idx], marker = '.')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel(r'$\Delta p$ (% of $|\vec{r_o}|$ and $|\vec{v_o}|$)')
    plt.legend()
    plt.title('Initial state difference from truth in RSW')

    print('\n')
    print(r'ERE - Initial state (cart):', str(100.0 * abs(parameter_evolution_array[-1, 1:7] / true_parameters[:6] - 1.0)))
    print(r'ERE - Initial state (rsw):', str(initial_state_errors[-1,:]))
    for idx, parameter in enumerate(estimated_parameters[1:]):
        print(r'ERE -', parameter + ':',
              str(100.0 * abs(parameter_evolution_array[-1, 7 + idx] / true_parameters[6 + idx] - 1.0)))

    # PARAMETER EVOLUTION AS CHANGES BETWEEN CONSECUTIVE ITERATIONS
    parameter_changes = np.zeros([number_of_iterations, len(parameter_evolution[0])])
    for k in range(number_of_iterations):
        parameter_changes[k,:] = 100.0*abs(parameter_evolution[k+1] / parameter_evolution[k] - 1.0)

    plt.figure()
    for idx in range(len(true_parameters)):
        plt.plot(parameter_evolution_array[1:,0], parameter_changes[:, idx], label = plot_legends[idx], marker = '.')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Parameter change [% of pre-fit]')
    plt.legend()
    plt.title('Parameter change between pre- and post-fit')

    # CORRELATION MATRIX
    plt.matshow(abs(estimation_output.correlations))
    plt.title('Correlation matrix (absolute values)')
    plt.xticks(list(range(len(plot_legends))), plot_legends, fontsize = 15)
    plt.yticks(list(range(len(plot_legends))), plot_legends, fontsize = 15)
    cb = plt.colorbar()
    cb.set_ticks(ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 15)

print('\nPROGRAM COMPLETED SUCCESSFULLY')
