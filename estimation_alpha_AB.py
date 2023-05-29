from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import estimation, estimation_setup, Estimator
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

save = True
post_process_in_this_file = True

observation_model = 'A1'
estimation_model = 'A1'

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
if observation_model == 'A1':
    trajectory_file = '/home/yorch/thesis/phobos-ephemerides-3500.txt'
elif observation_model == 'B':
    trajectory_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
else:
    raise ValueError('Invalid observation model selected.')

phobos_ephemerides = get_ephemeris_from_file(trajectory_file)
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, scaled_amplitude)

# PROPAGATOR
simulation_time = 300.0 * constants.JULIAN_DAY
initial_estimation_epoch = 1.0 * constants.JULIAN_YEAR
translational_perturbation = np.array([100.0, 200.0, 100.0])
# translational_perturbation = np.array([10.0, 2.0, 5.0])*1000.0
if estimation_model == 'A1':
    perturbation = np.zeros(6)
    perturbation[:3] = translational_perturbation
    perturbed_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch) + perturbation
    propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, initial_estimation_epoch, initial_state = perturbed_initial_state)
if estimation_model == 'B':
    perturbation = np.zeros(13)
    perturbation[:3] = translational_perturbation
    perturbed_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch) + perturbation
    propagator_settings = get_model_b_propagator_settings(bodies, simulation_time, initial_epoch = initial_estimation_epoch,
                                                          initial_state = perturbed_initial_state)

# PARAMETERS TO ESTIMATE
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Libration amplitude missing ???
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# LINK SET UP
link_ends = { observation.observed_body : observation.body_origin_link_end_id('Phobos') }
link = observation.LinkDefinition(link_ends)

# NOW, WE CREATE THE OBSERVATIONS
observation_model_settings = observation.cartesian_position(link)

t0 = 1.0 * constants.JULIAN_YEAR + 86400.0
tf = 1.0 * constants.JULIAN_YEAR + simulation_time
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
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = maximum_number_of_iterations)
estimation_input = estimation.EstimationInput(observation_collection, convergence_checker = convergence_checker)
# estimation_input.define_estimation_settings(save_state_history_per_iteration=True)

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

if save:

    # SAVE RESULTS
    save_dir = getcwd() + '/estimation-ab/alpha/'

    save2txt(residual_history, save_dir + 'residual-history-a1a1-test-far.txt')
    save2txt(parameter_evolution, save_dir + 'parameter-evolution-a1a1-test-far.txt')
    save2txt(residual_rms_evolution, save_dir + 'rms-evolution-a1a1-test-far.txt')


if post_process_in_this_file:

    residual_history_array = result2array(residual_history)
    parameter_evolution_array = result2array(parameter_evolution)
    rms_array = result2array(residual_rms_evolution)

    number_of_iterations = int(parameter_evolution_array.shape[0] - 1)
    true_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch)

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
    plt.plot(parameter_evolution_array[:,0], (parameter_evolution_array[:,1] - true_initial_state[0]) / 1000.0, label=r'$x_o$', marker='.')
    plt.plot(parameter_evolution_array[:,0], (parameter_evolution_array[:,2] - true_initial_state[1]) / 1000.0, label=r'$y_o$', marker='.')
    plt.plot(parameter_evolution_array[:,0], (parameter_evolution_array[:,3] - true_initial_state[2]) / 1000.0, label=r'$z_o$', marker='.')
    plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,4] - true_initial_state[3], label=r'$v_{x,o}$', marker='.')
    plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,5] - true_initial_state[4], label=r'$v_{y,o}$', marker='.')
    plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,6] - true_initial_state[5], label=r'$v_{z,o}$', marker='.')
    plt.grid()
    plt.xlabel('Iteration number')
    plt.ylabel('Parameter difference from truth [km | m/s]')
    plt.legend()
    plt.title('Parameter history')

    parameter_changes = np.zeros([number_of_iterations, 6])
    for k in range(number_of_iterations):
        parameter_changes[k, :] = parameter_evolution[k + 1] - parameter_evolution[k]

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
