import os

from Auxiliaries import *

########################################################################################################################
########################################################################################################################

post_process_single_estimation = False
post_process_estimation_duration_batch = True

run_dir = 'bravo/batch 2023-08-12 10:55:22.243514/estimation-time-75'
batch_dir = 'bravo/batch 2023-08-12 10:55:22.243514'

if 'alpha' in batch_dir:
    exclude_estimation_times = [90.0]
else:
    exclude_estimation_times = []

plot_cartesian_things = False

########################################################################################################################
########################################################################################################################

base_dir = os.getcwd() + '/estimation-results/'

if post_process_single_estimation:

    read_dir = os.getcwd() + '/estimation-results/' + run_dir + '/'

    print('Reading settings...')
    settings = EstimationSettings(read_dir + 'settings.log')
    ########################################################################################################################
    #   SETTINGS REQUIRED IN THIS FILE
    initial_estimation_epoch = settings.estimation_settings['initial estimation epoch']
    estimated_parameters = settings.estimation_settings['estimated parameters']
    norm_position_residuals = settings.estimation_settings['norm position residuals']
    convert_residuals_to_rsw = settings.estimation_settings['convert residuals to rsw']
    plot_observations = settings.postprocess_settings['plot observations']
    plot_cartesian_residuals = settings.postprocess_settings['plot cartesian residuals']
    plot_normed_residuals = settings.postprocess_settings['plot normed residuals']
    plot_rsw_residuals = settings.postprocess_settings['plot rsw residuals']
    ########################################################################################################################

    parameter_evolution = read_vector_history_from_file(read_dir + 'parameter-evolution.dat')
    true_initial_state = settings.get_initial_state('true')
    observation_history = read_vector_history_from_file(read_dir + 'observation-history.dat')
    residual_histories = [read_vector_history_from_file(read_dir + 'residual-history-cart.dat')]
    if norm_position_residuals:
        residual_histories.append(read_vector_history_from_file(read_dir + 'residual-history-norm.dat'))
    if convert_residuals_to_rsw:
        residual_histories.append(read_vector_history_from_file(read_dir + 'residual-history-rsw.dat'))
    residual_statistical_indicators_per_iteration = [read_vector_history_from_file(read_dir + 'residual-indicators-per-iteration-cart.dat')]
    if norm_position_residuals:
        residual_statistical_indicators_per_iteration.append(read_vector_history_from_file(read_dir + 'residual-indicators-per-iteration-norm.dat'))
    if convert_residuals_to_rsw:
        residual_statistical_indicators_per_iteration.append(read_vector_history_from_file(read_dir + 'residual-indicators-per-iteration-rsw.dat'))

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
            if k == 0: title = 'Pre-fit residual history'
            else: title = 'Post-fit residual history (iteration ' + str(k) + ')'
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
            plt.title(title)

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
            if k == 0: title = 'Pre-fit residual history'
            else: title = 'Post-fit residual history (iteration ' + str(k) + ')'
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
            plt.title(title)

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,1], marker='.', color='k', ls='--', label=r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,2], marker='.', color=colors[0], label=r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,3], marker='x', color=colors[1], label=r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,4], marker='.', color='k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'Residual [m]')
        plt.legend()
        plt.title('Post fit residuals - $R$ component')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,5], marker='.', color='k', ls='--', label=r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,6], marker='.', color=colors[0], label=r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,7], marker='x', color=colors[1], label=r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,8], marker='.', color='k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'Residual [m]')
        plt.legend()
        plt.title('Post fit residuals - $S$ component')

        plt.figure()
        plt.plot(indicators_array[:,0], indicators_array[:,9], marker='.', color='k', ls='--', label=r'Min/Max')
        plt.plot(indicators_array[:,0], indicators_array[:,10], marker='.', color=colors[0], label=r'Mean')
        plt.plot(indicators_array[:,0], indicators_array[:,11], marker='x', color=colors[1], label=r'RMS')
        plt.plot(indicators_array[:,0], indicators_array[:,12], marker='.', color='k', ls='--')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'Residual [m]')
        plt.legend()
        plt.title('Post fit residual - $W$ component')

        plt.figure()
        plt.semilogy(indicators_array[:,0], indicators_array[:,3], marker='.', label=r'R')
        plt.semilogy(indicators_array[:,0], indicators_array[:,7], marker='.', label=r'S')
        plt.semilogy(indicators_array[:,0], indicators_array[:,11], marker='.', label=r'W')
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel(r'RMS($\varepsilon_{e,i}$) [m]')
        plt.legend()
        plt.title('Post fit residual RMS')

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
        if 'v' in plot_legends[idx]:
            plt.semilogy(parameter_evolution_array[:,0], 100.0*abs(parameter_evolution_array[:,idx+1] / true_parameters[idx] - 1.0), label = plot_legends[idx], marker = 'x', ls = 'dashed')
        else:
            plt.semilogy(parameter_evolution_array[:,0], 100.0*abs(parameter_evolution_array[:,idx+1] / true_parameters[idx] - 1.0), label = plot_legends[idx], marker = '.')
        # plt.semilogy(parameter_evolution_array[:,0], 100.0*abs(parameter_evolution_array[:,idx+1] / true_parameters[idx] - 1.0), label = plot_legends[idx], marker = marker)
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
        if 'v' in current_legend_list[idx]:
            plt.semilogy(parameter_evolution_array[:,0], abs(initial_state_errors[:,idx]), label = current_legend_list[idx], marker = 'x', ls = 'dashed')
        else:
            plt.semilogy(parameter_evolution_array[:,0], abs(initial_state_errors[:,idx]), label = current_legend_list[idx], marker = '.')
        # plt.semilogy(parameter_evolution_array[:,0], abs(initial_state_errors[:,idx]), label = current_legend_list[idx], marker = marker)
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel(r'$\Delta\vec x_i$ (% of $|\vec{r_o}|$ and $|\vec{v_o}|$)')
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
        if 'v' in plot_legends[idx]:
            marker = 'x'
        else:
            marker = '.'
        plt.plot(parameter_evolution_array[1:,0], parameter_changes[:, idx], label = plot_legends[idx], marker = '.')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Parameter change [% of pre-fit]')
    plt.legend()
    plt.title('Parameter change between pre- and post-fit')

    # CORRELATION MATRIX
    number_of_parameters = len(parameter_evolution_array[0,1:])
    correlation_matrix = read_matrix_from_file(read_dir + 'correlation-matrix.cor', [number_of_parameters, number_of_parameters])
    plt.matshow(abs(correlation_matrix))
    plt.title('Correlation matrix (absolute values)')
    plt.xticks(list(range(len(plot_legends))), plot_legends, fontsize = 15)
    plt.yticks(list(range(len(plot_legends))), plot_legends, fontsize = 15)
    cb = plt.colorbar()
    cb.set_ticks(ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 15)

if post_process_estimation_duration_batch:

    ms = 15

    if 'batch' not in batch_dir:
        raise ValueError('Invalid batch directory.')

    print('Processing folder ' + batch_dir)

    batch_dir = base_dir + batch_dir + '/'
    # estimation_folders = os.listdir(batch_dir)
    # estimation_folders.remove('settings.log')

    print('Reading settings file...')
    settings = EstimationSettings(batch_dir + 'settings.log')
    ################################################################################################################
    #   SETTINGS REQUIRED IN THIS FILE
    initial_estimation_epoch = settings.estimation_settings['initial estimation epoch']
    arc_durations = settings.estimation_settings['duration of estimated arc']
    estimated_parameters = settings.estimation_settings['estimated parameters']
    norm_position_residuals = settings.estimation_settings['norm position residuals']
    convert_residuals_to_rsw = settings.estimation_settings['convert residuals to rsw']
    ################################################################################################################
    arc_durations = np.array([arc_durations[idx] for idx in range(len(arc_durations)) if arc_durations[idx] / 86400.0 not in exclude_estimation_times])
    estimation_folders = []
    for T in arc_durations:
        estimation_folders.append('estimation-time-' + str(int(T / 86400.0)))

    number_of_parameters = len(estimated_parameters)
    length_of_estimated_state_vector = settings.get_length_of_estimated_state()

    # STRUCTURE OF DATA MATRIX
    # T RT ITC PFRI ERE
    # T : Duration of estimated arc (1)
    # RT : Runtime (1)
    # ITC : Iterations to converge (1)
    # PFRI : Post-fit residuals indicators (4 x (3 cartesian + 1 if norm_residuals + 3 if convert_residuals_to_rsw))
    # ERE : Estimation result error (initial state (6 cartesian + 6 RSW) + 4 if estimation model is coupled + len(number of parameters) - 1)

    cols = int(20 + 4*(norm_position_residuals + 3*convert_residuals_to_rsw) + length_of_estimated_state_vector + len(estimated_parameters))
    data_matrix = np.zeros([len(estimation_folders), cols])
    data_matrix[:,0] = arc_durations

    # The true initial state (and therefore the RSW matrix referenced to it) is common to all runs.
    true_initial_state = settings.get_initial_state('true')
    R = inertial_to_rsw_rotation_matrix(true_initial_state[:6])
    full_state_rotation_matrix = np.concatenate(
        (np.concatenate((R, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), R), 1)), 0)
    r0 = norm(true_initial_state[:3])
    v0 = norm(true_initial_state[3:6])
    omega0 = norm(true_initial_state[9:])
    true_parameters = settings.get_true_parameters()

    for idx, estimation_dir in enumerate(estimation_folders):

        print('Processing estimation ' + str(idx+1) + '/' + str(len(estimation_folders)))

        current_dir = batch_dir + estimation_dir + '/'

        parameter_evolution = read_vector_history_from_file(current_dir + 'parameter-evolution.dat')
        # observation_history = read_vector_history_from_file(current_dir + 'observation-history.dat')
        # residual_histories = [read_vector_history_from_file(current_dir + 'residual-history-cart.dat')]
        # if norm_position_residuals:
        #     residual_histories.append(read_vector_history_from_file(current_dir + 'residual-history-norm.dat'))
        # if convert_residuals_to_rsw:
        #     residual_histories.append(read_vector_history_from_file(current_dir + 'residual-history-rsw.dat'))
        residual_statistical_indicators_per_iteration = [
            read_vector_history_from_file(current_dir + 'residual-indicators-per-iteration-cart.dat')]
        if norm_position_residuals:
            residual_statistical_indicators_per_iteration.append(
                read_vector_history_from_file(current_dir + 'residual-indicators-per-iteration-norm.dat'))
        if convert_residuals_to_rsw:
            residual_statistical_indicators_per_iteration.append(
                read_vector_history_from_file(current_dir + 'residual-indicators-per-iteration-rsw.dat'))

        parameter_evolution_array = result2array(parameter_evolution)
        parameter_evolution_array = settings.convert_libration_amplitude(parameter_evolution_array)
        number_of_iterations = int(parameter_evolution_array.shape[0] - 1)

        # Runtime and iterations-to-convergence columns
        with open(current_dir + 'log.txt', 'r') as file:
            lines = file.readlines()
            data_matrix[idx,1] = lines[-2].split(' ')[-1].replace('\n', '')
            data_matrix[idx,2] = lines[-1].split(' ')[-1].replace('\n', '')

        # RESIDUALS INDICATORS IN ALL EXISTING FORMATS (CART, NORM, RSW). Note: We need to keep shifting things to the right with the extras.
        data_matrix[idx, 3:15] = result2array(residual_statistical_indicators_per_iteration[0])[-1,1:].reshape([1,12])
        if norm_position_residuals:
            data_matrix[idx, 15:19] = result2array(residual_statistical_indicators_per_iteration[1])[-1,1:].reshape([1,4])
            extra_norm_res = 4
        else:
            extra_norm_res = 0

        if convert_residuals_to_rsw:
            data_matrix[idx, 15+extra_norm_res:27+extra_norm_res] = result2array(residual_statistical_indicators_per_iteration[-1])[-1,1:].reshape([1,12])
            extra_rsw_res = 12
        else:
            extra_rsw_res = 0

        # PARAMETER ERRORS
        # Initial state - Cartesian
        idx1 = 15 + extra_norm_res + extra_rsw_res
        idx2 = 21 + extra_norm_res + extra_rsw_res
        data_matrix[idx, idx1:idx2] = parameter_evolution_array[-1,1:7] - true_initial_state[:6]
        # Initial state - RSW
        idx1 = 21 + extra_norm_res + extra_rsw_res
        idx2 = 27 + extra_norm_res + extra_rsw_res
        data_matrix[idx, idx1:idx2] = full_state_rotation_matrix @ (parameter_evolution_array[-1,1:7] - true_initial_state[:6])
        if length_of_estimated_state_vector == 13:
            idx1 = 27 + extra_norm_res + extra_rsw_res
            idx2 = 34 + extra_norm_res + extra_rsw_res
            data_matrix[idx,idx1:idx2] = parameter_evolution_array[-1,7:11] - true_initial_state[6:]
            extra_rot = 7
        else:
            extra_rot = 0
        # Other parameters
        idx1 = 27 + extra_norm_res + extra_rsw_res + extra_rot
        for k in range(int(number_of_parameters - 1)):
            data_matrix[idx,idx1+k] = parameter_evolution_array[-1,7+k] - true_parameters[6+k]

    save_matrix_to_file(data_matrix, batch_dir + 'batch_analysis_matrix.dat')

    # PLOTS
    xlabel = 'Duration of estimated arc [days]'

    # # Runtimes
    # plt.figure()
    # plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,1], marker = '.')
    # plt.grid()
    # # plt.yscale('log')
    # plt.xlabel(xlabel)
    # plt.ylabel('RT [s]')
    # plt.title('Runtimes')
    #
    # # Iterations to convergence
    # plt.figure()
    # plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,2], marker = '.')
    # plt.grid()
    # plt.xlabel(xlabel)
    # plt.ylabel('ITC')
    # plt.title('Iterations to convergence')

    # PFRI
    if plot_cartesian_things:

        # Cartesian residuals
        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,3] * 1e2, marker = '.', color = 'k', ls = '--', label = 'Min/Max')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,4] * 1e2, marker = '.', label = 'Mean')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,5] * 1e2, marker = '.', label = 'RMS')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,6] * 1e2, marker = '.', color = 'k', ls = '--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\varepsilon_x$ [cm]')
        plt.title('Post-fit residuals indicators (x component)')

        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,7] * 1e2, marker = '.', color = 'k', ls = '--', label = 'Min/Max')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,8] * 1e2, marker = '.', label = 'Mean')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,9] * 1e2, marker = '.', label = 'RMS')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,10] * 1e2, marker = '.', color = 'k', ls = '--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\varepsilon_y$ [cm]')
        plt.title('Post-fit residuals indicators (y component)')

        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,11] * 1e2, marker = '.', color = 'k', ls = '--', label = 'Min/Max')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,12] * 1e2, marker = '.', label = 'Mean')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,13] * 1e2, marker = '.', label = 'RMS')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,14] * 1e2, marker = '.', color = 'k', ls = '--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\varepsilon_z$ [cm]')
        plt.title('Post-fit residuals indicators (z component)')

    if norm_position_residuals:

        # Normed residuals
        plt.figure()
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,15] * 1e2, marker = '.', color = 'k', ls = '--', label = 'Min/Max')
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,16] * 1e2, marker = '.', label = 'Mean')
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,17] * 1e2, marker = '.', label = 'RMS')
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,18] * 1e2, marker = '.', color = 'k', ls = '--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$|\vec\varepsilon|$ [cm]')
        plt.title('Post-fit residuals indicators (normed)')

    if convert_residuals_to_rsw:

        extra = extra_norm_res

        # RSW residuals
        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,15+extra] * 1e2, marker = '.', color = 'k', ls = '--', label = 'Min/Max')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,16+extra] * 1e2, marker = '.', label = 'Mean')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,17+extra] * 1e2, marker = '.', label = 'RMS')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,18+extra] * 1e2, marker = '.', color = 'k', ls = '--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\varepsilon_r$ [cm]')
        plt.title('Post-fit residuals indicators (R component)')

        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,19+extra] * 1e2, marker = '.', color = 'k', ls = '--', label = 'Min/Max')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,20+extra] * 1e2, marker = '.', label = 'Mean')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,21+extra] * 1e2, marker = '.', label = 'RMS')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,22+extra] * 1e2, marker = '.', color = 'k', ls = '--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\varepsilon_s$ [cm]')
        plt.title('Post-fit residuals indicators (S component)')

        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,23+extra] * 1e2, marker='.', color='k', ls='--', label='Min/Max')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,24+extra] * 1e2, marker='.', label='Mean')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,25+extra] * 1e2, marker='.', label='RMS')
        plt.plot(data_matrix[:,0] / 86400.0, data_matrix[:,26+extra] * 1e2, marker='.', color='k', ls='--')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\varepsilon_w$ [cm]')
        plt.title('Post-fit residuals indicators (W component)')

        plt.figure()
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,17+extra] * 1e2, marker = '.', label = r'$R$')
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,21+extra] * 1e2, marker = '.', label = r'$S$')
        plt.semilogy(data_matrix[:,0] / 86400.0, data_matrix[:,25+extra] * 1e2, marker = '.', label = r'$W$')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'RMS($\varepsilon_i$) [cm]')
        plt.title('Post-fit residuals RMS')

    # ERE
    if plot_cartesian_things:

        extra = extra_norm_res + extra_rsw_res

        # State vector (cartesian)
        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,15+extra]) / r0, marker = '.', markersize = ms, label = r'$\Delta x_o$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,16+extra]) / r0, marker = '.', markersize = ms, label = r'$\Delta y_o$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,17+extra]) / r0, marker = '.', markersize = ms, label = r'$\Delta z_o$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,18+extra]) / v0, marker = 'x', ls = 'dashed', label = r'$\Delta v_{x,o}$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,19+extra]) / v0, marker = 'x', ls = 'dashed', label = r'$\Delta v_{y,o}$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,20+extra]) / v0, marker = 'x', ls = 'dashed', label = r'$\Delta v_{z,o}$')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\Delta\vec x_{o,i}$ [% of $|\vec r_o|$ and $|\vec v_o|$]')
        plt.title('Error in estimated initial state w.r.t. truth')

    extra = extra_norm_res + extra_rsw_res

    # State vector (RSW)
    plt.figure()
    plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,21+extra]) / r0, marker = '.', markersize = ms, label = r'$\Delta R_o$')
    plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,22+extra]) / r0, marker = '.', markersize = ms, label = r'$\Delta S_o$')
    plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,23+extra]) / r0, marker = '.', markersize = ms, label = r'$\Delta W_o$')
    plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,24+extra]) / v0, marker = 'x', ls = 'dashed', label = r'$\Delta v_{r,o}$')
    plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,25+extra]) / v0, marker = 'x', ls = 'dashed', label = r'$\Delta v_{s,o}$')
    plt.plot(data_matrix[:,0] / 86400.0, 100.0 * abs(data_matrix[:,26+extra]) / v0, marker = 'x', ls = 'dashed', label = r'$\Delta v_{w,o}$')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(r'$\Delta\vec x_o$ [% of $|\vec r_o|$ and $|\vec v_o|$]')
    plt.title('Error in estimated initial state w.r.t. truth')

    if length_of_estimated_state_vector > 6:

        # State vector (rot)
        plt.figure()
        plt.plot(data_matrix[:,0] / 86400.0, abs(data_matrix[:,21+extra]), marker = '.', markersize = ms, label = r'$\Delta q_0$')
        plt.plot(data_matrix[:,0] / 86400.0, abs(data_matrix[:,22+extra]), marker = '.', markersize = ms, label = r'$\Delta q_1$')
        plt.plot(data_matrix[:,0] / 86400.0, abs(data_matrix[:,23+extra]), marker = '.', markersize = ms, label = r'$\Delta q_2$')
        plt.plot(data_matrix[:,0] / 86400.0, abs(data_matrix[:,24+extra]), marker = '.', markersize = ms, label = r'$\Delta q_3$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0*abs(data_matrix[:,25+extra]) / omega0, marker = 'x', ls = 'dashed', label = r'$\Delta\omega_1 [% of $|\vec\omega_o|$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0*abs(data_matrix[:,26+extra]) / omega0, marker = 'x', ls = 'dashed', label = r'$\Delta\omega_2 [% of $|\vec\omega_o|$')
        plt.plot(data_matrix[:,0] / 86400.0, 100.0*abs(data_matrix[:,27+extra]) / omega0, marker = 'x', ls = 'dashed', label = r'$\Delta\omega_3 [% of $|\vec\omega_o|$')
        plt.grid()
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\Delta\vec x_o$ [-]')
        plt.title('Error in estimated initial state w.r.t. truth')

    if len(estimated_parameters) > 1:

        extra = extra_norm_res + extra_rsw_res + extra_rot
        plot_legends = settings.get_plot_legends()
        plot_legends = [legend for legend in plot_legends if 'o' not in legend]
        plt.figure()
        for k in range(len(estimated_parameters)-1):
            temp_to_plot = 100.0 * abs(data_matrix[:,27+extra+k] / true_parameters[length_of_estimated_state_vector + k])
            plt.plot(data_matrix[:,0] / 86400.0, temp_to_plot, marker = '.', label = plot_legends[k])
        plt.grid()
        # plt.yscale('log')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(r'$\Delta p$ [% of truth]')
        plt.title('Error in estimated parameters')


print('\nPROGRAM COMPLETED SUCCESSFULLY')
