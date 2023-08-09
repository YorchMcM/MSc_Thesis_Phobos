from Auxiliaries import *

read_dir = os.getcwd() + '/estimation-results/alpha-1/2023-07-20 16:32:38.289147/'

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
number_of_parameters = len(parameter_evolution_array[0,1:])
correlation_matrix = read_matrix_from_file(read_dir + 'correlation-matrix.cor', [number_of_parameters, number_of_parameters])
plt.matshow(abs(correlation_matrix))
plt.title('Correlation matrix (absolute values)')
plt.xticks(list(range(len(plot_legends))), plot_legends, fontsize = 15)
plt.yticks(list(range(len(plot_legends))), plot_legends, fontsize = 15)
cb = plt.colorbar()
cb.set_ticks(ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 15)

print('\nPROGRAM COMPLETED SUCCESSFULLY')