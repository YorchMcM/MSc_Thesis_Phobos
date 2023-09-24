'''

In this script we will define model A2. It propagates the rotational dynamics ALONE.

Note: the elements marked with an asterisk (*) are partially or fully defined in this script and are regarded as
something close to "user inputs". The others are fully set somewhere in the Auxiliaries module.

ENVIRONMENT
· Global frame origin: Mars' center of mass
· Global frame orientation: Earth's equator of J2000
* Ephemeris model: Tabulated states propagated by model A1.
· Mars' gravity field: default from Tudat
· Phobos' gravity field: From Le Maistre (2019) - Only coefficients C00, C20 and C22.
· Phobos' inertia tensor: Derived from the harmonic coefficients.
· Ephemeris and gravitational parameters of all other bodies: defaults from Tudat

TORQUES
· Center-of-mass  to  Phobos' quadrupole gravity field of the following bodies: Mars, Sun, Earth, Deimos, Jupiter

PROPAGATOR
· Propagator: Quaternion and angular velocity vector
* Initial epoch: 02/01/2000 at 12:00:00
· Initial state: Tudat-generated damped initial state
* Simulation time: 10 times the largest dissipation time

INTEGRATOR
· Integrator: fixed-step RKDP7(8) with a fixed time step of 5 minutes

The propagation in model A1 gives an average mean motion of 2.278563609852602e-4 rad/s = 19.68678958912648 rad/day. The
associated orbital period is of 7h 39min 35.20s.
The tweaked rotational motion in this model is of 2.28035245e-4 rad/s = 19.702245168 rad/day. The associated rotational
period is of 7h 39min 13.57s.

'''

from Auxiliaries import *

########################################################################################################################
# SETTINGS

# Execution
verbose = True
retrieve_dependent_variables = True
save = False
simulate_and_save_full_dynamics = False
generate_ephemeris_file = True
check_undamped = True
checks = [0, 0, 0, 0, 0, 0]

########################################################################################################################

if sum(checks) > 0:
    retrieve_dependent_variables = True

#                                  4h,  8h,  16h,  1d 8h, 2d 16h, 5d 8h, 10d 16h, 21d 8h, 42d 16h, 85d 8h, 170d 16h, 341d 8h // Up to 3143d 8h in get_zero_proper_mode function
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0,  64.0,   128.0, 256.0,   512.0,  1024.0,  2048.0, 4096.0,   8192.0])*3600.0)  # In seconds.
# dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0])*3600.0)  # In seconds.

if save or simulate_and_save_full_dynamics:
    save_dir = os.getcwd() + '/simulation-results/model-a2/' + str(datetime.now()) + '/'
    os.makedirs(save_dir)


# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
if verbose:
    print('Creating universe...')
ephemeris_file = 'ephemeris/translation-a.eph'
ephemeris_file = 'simulation-results/model-a1/2023-07-13 10:24:42.226993/state-history.dat'
bodies = get_solar_system('A2', ephemeris_file)


# DEFINE PROPAGATION
if verbose: print('Setting up propagation...')
initial_epoch = 86400.0
initial_state = get_undamped_initial_state_at_epoch(bodies, 'A2', initial_epoch, phobos_mean_rotational_rate)
simulation_time = 10.0 * dissipation_times[-1]
if retrieve_dependent_variables: dependent_variables = get_list_of_dependent_variables('A2', bodies)
else: dependent_variables = []
propagator_settings = get_propagator_settings('A2', bodies, initial_epoch, initial_state, simulation_time, dependent_variables)


# SIMULATE DYNAMICS BACK AND FORTH AND OBTAIN DAMPED INITIAL STATE TOGETHER WITH A WHOLE BUNCH OF OTHER THINGS
print('Simulating dynamics. Going into the depths of Tudat...')
tic = time()
damping_results = numerical_simulation.propagation.get_damped_proper_mode_initial_rotational_state(bodies,
                                                                                                   propagator_settings,
                                                                                                   phobos_mean_rotational_rate,
                                                                                                   dissipation_times)
tac = time()
if verbose: print('SIMULATIONS FINISHED. Time taken:', (tac-tic) / 60.0, 'minutes.')


# SAVE RESULTS
if save:
    if verbose: print('Saving results...')
    log = '\n· Initial epoch: ' + str(initial_epoch) + ' seconds\n· Simulation time: ' + \
          str(simulation_time / constants.JULIAN_DAY) + ' days\n· Damping times: ' + str(dissipation_times) + '\n'
    save_initial_states(damping_results, save_dir + 'initial_states.dat')
    with open(save_dir + 'log.log', 'w') as file: file.write(log)
    save2txt(damping_results.forward_backward_states[0][0], save_dir + 'states-undamped.dat')
    if retrieve_dependent_variables:
        save2txt(damping_results.forward_backward_dependent_variables[0][0], save_dir + 'dependents-undamped.dat')

    for idx, current_damping_time in enumerate(dissipation_times):
        time_str = str(int(current_damping_time / 3600.0))
        save2txt(damping_results.forward_backward_states[idx+1][1], save_dir + 'states-d' + time_str + '.dat')
        if retrieve_dependent_variables:
            save2txt(damping_results.forward_backward_dependent_variables[idx+1][1], save_dir + 'dependents-d' + time_str + '.dat')
    if verbose: print('Results saved.')


# SIMULATE AND SAVE THE RESULTS OF ALL DAMPING TIMES UP TO THE SAME FINAL EPOCH
if simulate_and_save_full_dynamics:
    if verbose: print('Simulating and saving full dynamics for all damping times...')
    tic = time()
    time_str = str(int(dissipation_times[-1] / 3600.0))
    save2txt(damping_results.forward_backward_states[-1][1], save_dir + 'states-d' + time_str + '-full.dat')
    if retrieve_dependent_variables:
        save2txt(damping_results.forward_backward_dependent_variables[-1][1], save_dir + 'dependents-d' + time_str +
                 '-full.dat')
    for idx, current_damping_time in enumerate(dissipation_times[:-1]):
        time_str = str(int(current_damping_time / 3600.0))
        print('Simulation ' + str(idx+1) + '/' + str(len(dissipation_times[:-1])))
        current_initial_epoch = list(damping_results.forward_backward_states[idx+1][1].keys())[-1]
        current_initial_state = damping_results.forward_backward_states[idx+1][1][current_initial_epoch]
        current_simulation_time = initial_epoch + simulation_time - current_initial_epoch
        current_propagator_settings = get_propagator_settings('A2', bodies, current_initial_epoch, current_initial_state, current_simulation_time, dependent_variables)
        current_simulator = numerical_simulation.create_dynamics_simulator(bodies, current_propagator_settings)
        full_state_history = damping_results.forward_backward_states[idx+1][1] | current_simulator.state_history
        save2txt(full_state_history, save_dir + 'states-d' + time_str + '-full.dat')
        if retrieve_dependent_variables:
            full_dependent_variable_history = damping_results.forward_backward_dependent_variables[idx+1][1] | current_simulator.dependent_variable_history
            save2txt(full_dependent_variable_history, save_dir + 'dependents-d' + time_str + '-full.dat')
    tac = time()
    if verbose: print('SIMULATIONS FINISHED. Time taken:', (tac-tic) / 60.0, 'minutes.')


# GENERATE EPHEMERIS FILE
if generate_ephemeris_file:
    if verbose: print('Generating ephemeris file...')
    # if not simulate_and_save_full_dynamics:
    #     ephemeris_initial_epoch = list(damping_results.forward_backward_states[-1][1].keys())[-1]
    #     ephemeris_initial_state = damping_results.forward_backward_states[-1][1][ephemeris_initial_epoch]
    #     ephemeris_simulation_time = initial_epoch + simulation_time - ephemeris_initial_epoch
    #     ephemeris_propagator_settings = get_propagator_settings('A2', bodies, ephemeris_initial_epoch, ephemeris_initial_state, ephemeris_simulation_time, dependent_variables)
    #     ephemeris_simulator = numerical_simulation.create_dynamics_simulator(bodies, ephemeris_propagator_settings)
    #     ephemeris_state_history = damping_results.forward_backward_states[-1][1] | ephemeris_simulator.state_history
    # else: ephemeris_history = full_state_history
    ephemeris_history = damping_results.forward_backward_states[-1][1]
    eph_dir = os.getcwd() + '/ephemeris/'
    save2txt(ephemeris_history, eph_dir + 'rotation-a.eph')
    if retrieve_dependent_variables:
        save2txt(damping_results.forward_backward_dependent_variables[-1][1], eph_dir + '/associated-dependents/a2.dat')


# POST PROCESS / CHECKS - THIS IS ONLY POSSIBLE IF THE APPROPRIATE DEPENDENT VARIABLES ARE RETRIEVED.
if retrieve_dependent_variables:
    run_model_a2_checks(checks, bodies, damping_results, check_undamped)


print('PROGRAM COMPLETED SUCCESSFULLY')
