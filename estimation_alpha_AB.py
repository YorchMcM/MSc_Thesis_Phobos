from Auxiliaries import *

from tudatpy.kernel.numerical_simulation import estimation, estimation_setup, Estimator
from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
trajectory_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
imposed_trajectory = extract_elements_from_history(read_vector_history_from_file(trajectory_file), [0, 1, 2, 3, 4, 5])
phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source, scaled_amplitude)

# PROPAGATOR
simulation_time = 3.0 * constants.JULIAN_YEAR
initial_estimation_epoch = 1.0 * constants.JULIAN_YEAR
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, initial_epoch = initial_estimation_epoch)

# PARAMETERS TO ESTIMATE
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Libration amplitude missing ???
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# LINK SET UP
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
estimator = Estimator(
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

# number_of_iterations = estimation_output.residual_history.shape[1]
residual_history, parameter_evolution, residual_rms_evolution = extract_estimation_output(estimation_output, list(observation_times), 'position')

save2txt(residual_history, save_dir + 'residual-history.txt')
save2txt(parameter_evolution, save_dir + 'parameter-evolution.txt')
save2txt(residual_rms_evolution, save_dir + 'rms-evolution.txt')