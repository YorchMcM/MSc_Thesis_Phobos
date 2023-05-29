'''

Here we will check whether the partials with respect to the initial state are correctly computed when a synchronous
rotation model is used, which adds complexity when evaluating the derivatives. For this, we will be comparing the state
transition matrix that is computed by Tudat with a derivate computed through central partial differences. For this, the
variational equations will be propagated for a nominal trajectory and the state transition matrix will be saved. Then,
the initial state will be perturbed and the equations of motion will be re-simulated so that the finite difference can
be computed.

We will use the SingleArcVariationalSimulator instead of running the whole estimation. To create it, we need:

· Bodies
· Integrator settings
· Propagator settings
· Estimated parameters

We will create them one by one. We will only be propagating the equations of motion and the state transition matrix part
of the variational equations, both with the same propagator settings. Note that no observations will be needed and the
translational motion of Phobos will be numerically integrated. This means that the ephemeris model we set for Phobos is
completely irrelevant, but the rotational model is not. Here we will try three different rotational models: one without
libration, one with libration of amplitude 0.0º, and one with libration of amplitude of 1.1º.

We want to check the libration partials. Thus, we will keep the dynamical model as basic as possible to include only
this effect. Note that the only acceleration in which Phobos' orientation plays a role in in Phobos' own gravitational
attraction. There is no need to include anything else. This is the summary of the simulation:

ENVIRONMENT
· Global frame origin: Mars' center of mass
· Global frame orientation: Earth's equator of J2000
· Rotational model:
    - Synchronous
    - Synchronous + once-per-orbit longitudinal libration of amplitude 0.0º
    - Synchronous + once-per-orbit longitudinal libration of amplitude 1.1º (Rambaux et al. 2012)
· Mars' gravity field: default from Tudat
· Mars' rotational model: default from Tudat
· Phobos' gravity field: From Le Maistre (2019)

ACCELERATIONS
· Phobos' quadrupole gravity field (C20 & C22).

PROPAGATOR
· Propagator: Cowell
· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: from spice at initial epoch.
· Simulation time: 3 years (will play with this, though)

INTEGRATOR
· Integrator: fixed-step RKDP7(8) with a fixed time step of 5 minutes

'''

from Auxiliaries import *
from tudatpy.kernel.numerical_simulation import estimation_setup

cases = ['no libration', '0 libration', 'libration']
perturbations = np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
pert_strings = [r'$x_o$', r'$y_o$', r'$z_o$', r'$v_{x,o}$', r'$v_{y,o}$', r'$v_{z,o}$']

for case in cases:

    print('CASE:', case)

    # WE CREATE THE BODIES.

    # Mars
    bodies_to_create = ["Mars"]
    global_frame_origin = "Mars"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)
    # Phobos
    body_settings.add_empty_settings('Phobos')
    # Ephemeris and rotation models.
    body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
    body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
    # Gravity field.
    body_settings.get('Phobos').gravity_field_settings = get_gravitational_field('Phobos_body_fixed', 'QUAD', 'Le Maistre')
    bodies = environment_setup.create_system_of_bodies(body_settings)

    if case == '0 libration':
        bodies.get('Phobos').rotation_model.libration_calculator = \
            numerical_simulation.environment.DirectLongitudeLibrationCalculator(0.0)
    if case == 'libration':
        libration_amplitude = 1.1  # In degrees
        ecc_scale = 0.015034167790105173
        scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
        bodies.get('Phobos').rotation_model.libration_calculator = \
            numerical_simulation.environment.DirectLongitudeLibrationCalculator(scaled_amplitude)

    # NEXT UP ARE THE INTEGRATION SETTINGS

    coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    time_step = 300.0
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                      coefficients,
                                                                                      time_step,
                                                                                      time_step,
                                                                                      np.inf, np.inf)

    # THEN WE HAVE THE PROPAGATOR SETTINGS

    simulation_time = 3.0 * constants.JULIAN_YEAR

    acceleration_settings = {
        'Phobos' : dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(2, 2, 2, 2)] )
    }
    acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, ['Phobos'], ['Mars'])
    initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'NONE', 0.0)
    termination_condition = propagation_setup.propagator.time_termination(simulation_time, True)
    propagator_settings = propagation_setup.propagator.translational(['Mars'],
                                                                     acceleration_model,
                                                                     ['Phobos'],
                                                                     initial_state,
                                                                     0.0,
                                                                     integrator_settings,
                                                                     termination_condition)

    # AND FINALLY WE GET TO THE ESTIMATED PARAMETERS
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

    # AND FINALLY WE CREATE THE VARIATIONAL SIMULATOR
    variational_simulator = numerical_simulation.SingleArcVariationalSimulator( bodies,
                                                                                integrator_settings,
                                                                                propagator_settings,
                                                                                parameters_to_estimate,
                                                                                print_dependent_variable_data = False)
    STM_history = variational_simulator.state_transition_matrix_history
    derivative_errors = dict.fromkeys(list(STM_history.keys()))
    for key in list(derivative_errors.keys()): derivative_errors[key] = np.zeros([6,6])

    for idx, perturbation in enumerate(perturbations):

        print('Perturbing ' + pert_strings[idx] + 'component ...')
        diff = 2*sum(perturbation)

        new_initial_state = initial_state + perturbation
        propagator_settings = propagation_setup.propagator.translational(['Mars'],
                                                                         acceleration_model,
                                                                         ['Phobos'],
                                                                         new_initial_state,
                                                                         0.0,
                                                                         integrator_settings,
                                                                         termination_condition)
        simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        state_history_plus = simulator.state_history

        new_initial_state = initial_state - perturbation
        propagator_settings = propagation_setup.propagator.translational(['Mars'],
                                                                         acceleration_model,
                                                                         ['Phobos'],
                                                                         new_initial_state,
                                                                         0.0,
                                                                         integrator_settings,
                                                                         termination_condition)
        simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        state_history_minus = simulator.state_history

        array_to_plot = np.zeros([len(list(derivative_errors.keys())), 7])
        for epoch_idx, epoch in enumerate(list(derivative_errors.keys())):
            derivative_errors[epoch][:,idx] = abs(STM_history[idx][:,idx] -
                                                  (state_history_plus - state_history_minus) / diff)
            array_to_plot[epoch_idx,1:] = derivative_errors[epoch][:,idx]


        title = 'Partials w.r.t.' + pert_strings[idx]
        x_label = 'Time since J2000 [days]'
        y_label = r'$|\bar{H}_{i,' + str(idx) + r'} - |\tilde{H}_{i,' + str(idx) + r'}|$'
