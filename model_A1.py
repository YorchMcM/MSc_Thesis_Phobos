'''

In this script we will define model A1. It propagates the translational dynamics ALONE.

Note: the elements marked with an asterisk (*) are partially or fully defined in this script and are regarded as
something close to "user inputs". The others are fully set somewhere in the Auxiliaries module.

ENVIRONMENT
· Global frame origin: Mars' center of mass
· Global frame orientation: Earth's equator of J2000
* Rotational model (two possibilities):
    - Fully synchronous
    - Synchronous + once-per-orbit longitudinal libration of amplitude 1.1º (Rambaux et al. 2012)
· Mars' gravity field: default from Tudat
· Phobos' gravity field: From Le Maistre (2019) - Only coefficients C00, C20 and C22.
· Phobos' inertia tensor: Derived from the harmonic coefficients.
· Ephemeris and gravitational parameters of all other bodies: defaults from Tudat

ACCELERATIONS
· Mars' harmonic coefficients up to degree and order 12.
· Phobos' quadrupole gravity field (C20 & C22).
· Third-body point-mass forces by the Sun, Earth, Deimos and Jupiter

PROPAGATOR
· Propagator: Cowell
* Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: from spice at initial epoch.
* Simulation time: 3500 days

INTEGRATOR
· Integrator: fixed-step RKDP7(8) with a fixed time step of 5 minutes

'''

from Auxiliaries import *

########################################################################################################################
# SETTINGS

# Dynamics
include_libration = True

# Execution
verbose = True
retrieve_dependent_variables = False
save = False
generate_ephemeris_file = False
checks = [1, 1, 1, 1, 1]

########################################################################################################################

if sum(checks) > 0:
    retrieve_dependent_variables = True

if save:
    save_dir = os.getcwd() + '/simulation-results/model-a1/' + str(datetime.now()) + '/'
    os.makedirs(save_dir)

if include_libration: model = 'A1'
else: model = 'S'


# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
if verbose: print('Creating universe...')
bodies = get_solar_system(model)


# DEFINE PROPAGATION
if verbose: print('Setting up propagation...')
initial_epoch = 0.0
initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'None', 0.0)
simulation_time = 3500.0*constants.JULIAN_DAY
if retrieve_dependent_variables: dependent_variables = get_list_of_dependent_variables(model, bodies)
else: dependent_variables = []
propagator_settings = get_propagator_settings(model, bodies, initial_epoch, initial_state, simulation_time, dependent_variables)


# SIMULATE DYNAMICS
if verbose: print('Simulating dynamics...')
tic = time()
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
tac = time()
if verbose: print('SIMULATIONS FINISHED. Time taken:', (tac-tic) / 60.0, 'minutes.')


# SAVE RESULTS
if save:

    if verbose: print('Saving results...')
    log = '\n· Model: ' + model + '\n· Initial epoch: ' + str(initial_epoch) + ' seconds\n· Simulation time: ' + \
          str(simulation_time / constants.JULIAN_DAY) + ' days\n'
    with open(save_dir + 'log.log', 'w') as file: file.write(log)
    save2txt(simulator.state_history, save_dir + 'state-history.dat')
    if retrieve_dependent_variables:
        save2txt(simulator.dependent_variable_history, save_dir + 'dependent-variable-history.dat')
    if verbose: print('Results saved.')


# GENERATE EPHEMERIS FILE
if generate_ephemeris_file:

    if verbose: print('Generating ephemeris file...')
    eph_dir = os.getcwd() + '/ephemeris/'
    if model == 'S': filename = 'translational-s.eph'
    else: filename = 'translational-a.eph'
    save2txt(simulator.state_history, eph_dir + filename)


# POST PROCESS / CHECKS
if retrieve_dependent_variables:
    # run_model_a1_checks(checks, bodies, simulator)
    run_model_a1_checks([0, 0, 0, 1, 0], bodies, simulator)


print('PROGRAM FINISHED SUCCESSFULLY')