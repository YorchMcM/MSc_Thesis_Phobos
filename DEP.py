def inertia_tensor_from_spherical_harmonic_gravity_field(
        gravity_field: numerical_simulation.environment.SphericalHarmonicsGravityField) -> np.ndarray:

    '''
    This function is completely equivalent to the tudat-provided getInertiaTensor function defined in the file
    tudat.astro.gravitation.sphericalHarmonicsGravityField.cpp, while requiring less inputs.
    NOTE: Phobos' mean moment of inertia is hard-coded inside this function.
    '''

    try:
        C_20 = gravity_field.cosine_coefficients[2,0]
        C_21 = gravity_field.cosine_coefficients[2,1]
        C_22 = gravity_field.cosine_coefficients[2,2]
        S_21 = gravity_field.sine_coefficients[2,1]
        S_22 = gravity_field.sine_coefficients[2,2]
    except:
        raise ValueError('Insufficient spherical harmonics for the computation of an inertia tensor.')

    R = gravity_field.reference_radius
    M = gravity_field.gravitational_parameter / constants.GRAVITATIONAL_CONSTANT

    N_20 = get_normalization_constant(2, 0)
    N_21 = get_normalization_constant(2, 1)
    N_22 = get_normalization_constant(2, 2)

    C_20 = C_20 * N_20
    C_21 = C_21 * N_21
    C_22 = C_22 * N_22
    S_21 = S_21 * N_21
    S_22 = S_22 * N_22

    I = 0.43  # Mean moment of inertia taken from Rambaux 2012 (no other number found anywhere else)
    I_xx = C_20/3 - 2*C_22 + I
    I_yy = C_20/3 + 2*C_22 + I
    I_zz = -(2.0/3.0)*C_20 + I
    I_xy = -2.0*S_22
    I_xz = -C_21
    I_yz = -S_21

    inertia_tensor = (M*R**2) * np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

    return inertia_tensor


def get_model_a1_propagator_settings(bodies: numerical_simulation.environment.SystemOfBodies,
                                     simulation_time: float,
                                     initial_epoch: float = 0.0,
                                     initial_state: np.ndarray = None,
                                     dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [])\
        -> propagation_setup.propagator.TranslationalStatePropagatorSettings:

    '''
    This function will create the propagator settings for model A1. The simulation time is given as an input. The bodies,
    with their environment properties (ephemeris model/rotation model/gravity field/...) is also give as input. This
    function defines the following about the accelerations and integration:

    · Initial epoch: J2000 (01/01/2000 at 12:00)
    · Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
    · Accelerations:
        - Mars' harmonic coefficients up to degree and order 12.
        - Phobos' quadrupole gravity field (C20 & C22).
        - Third-body point-mass forces by the Sun, Earth, Moon, Deimos, Jupiter and Saturn (Moon and Saturn we'll see)
    · Propagator: Cartesian states

    :param bodies: The SystemOfBodies object of the simulation.
    :param simulation_time: The duration of the simulation.
    :param initial_epoch:
    :param initial_state:
    :param dependent_variables: The list of dependent variables to save during propagation.
    :return: propagator_settings
    '''

    bodies_to_propagate = ['Phobos']
    central_bodies = ['Mars']

    # ACCELERATION SETTINGS
    third_body_force = propagation_setup.acceleration.point_mass_gravity()
    acceleration_settings_on_phobos = dict(Mars=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)],
                                           Sun=[third_body_force],
                                           Earth=[third_body_force],
                                           # Moon=[third_body_force],
                                           Deimos=[third_body_force],
                                           Jupiter=[third_body_force],
                                           # Saturn=[third_body_force]
                                           )
    acceleration_settings = {'Phobos': acceleration_settings_on_phobos}
    acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)
    # INTEGRATOR
    time_step = 300.0  # These are 300s = 5min
    coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                      coefficients,
                                                                                      time_step,
                                                                                      time_step,
                                                                                      np.inf, np.inf)
    # PROPAGATION SETTINGS
    # Initial conditions
    if initial_state is None: initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'NONE', initial_epoch)
    # Termination condition
    termination_condition = propagation_setup.propagator.time_termination(initial_epoch + simulation_time, True)
    # The settings object
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_model,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     initial_epoch,
                                                                     integrator_settings,
                                                                     termination_condition,
                                                                     output_variables = dependent_variables)

    return propagator_settings


def get_model_a2_propagator_settings(bodies: numerical_simulation.environment.SystemOfBodies,
                                     simulation_time: float,
                                     initial_epoch: float,
                                     initial_state: np.ndarray,
                                     dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [],
                                     return_integrator_settings = False) \
        -> propagation_setup.propagator.TranslationalStatePropagatorSettings:

    '''
    This function will create the propagator settings for model A1. The simulation time is given as an input. The bodies,
    with their environment properties (ephemeris model/rotation model/gravity field/...) is also give as input. This
    function defines the following about the accelerations and integration:

    · Initial epoch: 01/01/2000 at 15:37:15 (first periapsis passage)
    · Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
    · Torques:
        - Mars' center of mass on Phobos' quadrupole field.
        - Center-of-mass-on-quadrople-field torques of the following bodies:
            Sun, Earth, Moon, Deimos, Jupiter and Saturn (Moon and Saturn we'll see)
    · Propagator: Cartesian states

    :param bodies: The SystemOfBodies object of the simulation.
    :param simulation_time: The duration of the simulation.
    :param dependent_variables: The list of dependent variables to save during propagation.
    :return: propagator_settings
    '''

    if bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[-1] < initial_epoch + simulation_time:
        warnings.warn('(get_model_a2_propagator_settings): Simulation time extends beyond provided ephemeris.')

    bodies_to_propagate = ['Phobos']

    # TORQUE SETTINGS
    torque_on_phobos = propagation_setup.torque.spherical_harmonic_gravitational(2,2)
    torque_settings_on_phobos = dict(Mars=[torque_on_phobos],
                                     Sun=[torque_on_phobos],
                                     Earth=[torque_on_phobos],
                                     Deimos=[torque_on_phobos],
                                     Jupiter=[torque_on_phobos]
                                     )
    torque_settings = {'Phobos': torque_settings_on_phobos}
    torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

    # INTEGRATOR
    time_step = 300.0  # These are 300s = 5min / 450s = 7.5min
    coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                      coefficients,
                                                                                      time_step,
                                                                                      time_step,
                                                                                      np.inf, np.inf)
    # PROPAGATION SETTINGS
    # Termination condition
    termination_condition = propagation_setup.propagator.time_termination(initial_epoch + simulation_time, True)
    # The settings object
    propagator_settings = propagation_setup.propagator.rotational(torque_model,
                                                                  bodies_to_propagate,
                                                                  initial_state,
                                                                  initial_epoch,
                                                                  integrator_settings,
                                                                  termination_condition,
                                                                  output_variables = dependent_variables)

    if return_integrator_settings: return propagator_settings, integrator_settings
    else: return propagator_settings


def get_model_b_propagator_settings(bodies: numerical_simulation.environment.SystemOfBodies,
                                    simulation_time: float,
                                    initial_epoch: float,
                                    initial_state: np.ndarray,
                                    dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [])\
        -> propagation_setup.propagator.TranslationalStatePropagatorSettings:

    bodies_to_propagate = ['Phobos']
    central_bodies = ['Mars']

    # ACCELERATION SETTINGS
    third_body_force = propagation_setup.acceleration.point_mass_gravity()
    acceleration_settings_on_phobos = dict(Mars=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)],
                                           Sun=[third_body_force],
                                           Earth=[third_body_force],
                                           # Moon=[third_body_force],
                                           Deimos=[third_body_force],
                                           Jupiter=[third_body_force],
                                           # Saturn=[third_body_force]
                                           )
    acceleration_settings = {'Phobos': acceleration_settings_on_phobos}
    acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings,
                                                                      bodies_to_propagate, central_bodies)

    # TORQUE SETTINGS
    torque_on_phobos = propagation_setup.torque.spherical_harmonic_gravitational(2, 2)
    torque_settings_on_phobos = dict(Mars=[torque_on_phobos],
                                     Sun=[torque_on_phobos],
                                     Earth=[torque_on_phobos],
                                     # Moon=[torque_on_phobos],
                                     Deimos=[torque_on_phobos],
                                     Jupiter=[torque_on_phobos]
                                     # Saturn=[torque_on_phobos]
                                     )
    torque_settings = {'Phobos': torque_settings_on_phobos}
    torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

    # INTEGRATOR
    time_step = 300.0  # These are 300s = 5min
    coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                      coefficients,
                                                                                      time_step,
                                                                                      time_step,
                                                                                      np.inf, np.inf)

    # GENERAL PROPAGATION SETTINGS
    # Termination condition
    termination_condition = propagation_setup.propagator.time_termination(simulation_time)

    # TRANSLATIONAL PROPAGATOR
    initial_translational_state = initial_state[:6]
    propagator_settings_trans = propagation_setup.propagator.translational(central_bodies,
                                                                           acceleration_model,
                                                                           bodies_to_propagate,
                                                                           initial_translational_state,
                                                                           initial_epoch,
                                                                           integrator_settings,
                                                                           termination_condition)

    # ROTATIONAL PROPAGATOR
    initial_rotational_state = initial_state[6:]
    propagator_settings_rot = propagation_setup.propagator.rotational(torque_model,
                                                                      bodies_to_propagate,
                                                                      initial_rotational_state,
                                                                      initial_epoch,
                                                                      integrator_settings,
                                                                      termination_condition)

    # MULTI-TYPE PROPAGATOR
    propagator_list = [propagator_settings_trans, propagator_settings_rot]
    propagator_settings = propagation_setup.propagator.multitype(propagator_list,
                                                                 integrator_settings,
                                                                 initial_epoch,
                                                                 termination_condition,
                                                                 output_variables = dependent_variables)

    return propagator_settings


def get_fake_initial_state(bodies: numerical_simulation.environment.SystemOfBodies,
                           initial_epoch: float,
                           omega: float) -> np.ndarray:

    initial_translational_state = bodies.get('Phobos').state_in_base_frame_from_ephemeris(initial_epoch)
    rtn_to_inertial_matrx = inertial_to_rsw_rotation_matrix(initial_translational_state).T
    initial_rotation_matrix = rtn_to_inertial_matrx.copy()
    initial_rotation_matrix[:,:2] = -initial_rotation_matrix[:,:2]
    initial_orientation_quaternion = mat2quat(initial_rotation_matrix)
    initial_angular_velocity = np.zeros(3)
    initial_angular_velocity[-1] = omega
    initial_state = np.concatenate((initial_orientation_quaternion, initial_angular_velocity))

    return initial_state


def run_environment_compatibility_diagnostics(self, bodies: numerical_simulation.environment.SystemOfBodies) -> None:

    initial_epoch = self.estimation_settings['initial estimation epoch']
    simulation_time = self.estimation_settings['duration of estimated arc']
    obs = self.observation_settings['observation type']
    out_of_bounds = False
    interpolation = False
    if type(bodies.get('Phobos').ephemeris) is numerical_simulation.environment.TabulatedEphemeris and obs == 'position':
        out_of_bounds = out_of_bounds or (initial_epoch == bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[0] > \
                        initial_epoch or bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[-1] < \
                        initial_epoch + simulation_time)
        interpolation = interpolation or (initial_epoch == bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[3] > \
                        initial_epoch or bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[-4] < \
                        initial_epoch + simulation_time)

    if type(bodies.get('Phobos').rotation_model) is numerical_simulation.environment.TabulatedRotationalEphemeris and obs == 'orientation':
        out_of_bounds = out_of_bounds or (initial_epoch == bodies.get('Phobos').rotation_model.interpolator.get_independent_values()[0] > \
                        initial_epoch or bodies.get('Phobos').rotation_model.interpolator.get_independent_values()[-1] < \
                        initial_epoch + simulation_time)
        interpolation = interpolation or (initial_epoch == bodies.get('Phobos').rotation_model.interpolator.get_independent_values()[3] > \
                        initial_epoch or bodies.get('Phobos').rotation_model.interpolator.get_independent_values()[-4] < \
                        initial_epoch + simulation_time)
    if out_of_bounds:
        warnings.warn(
            'The available ephemeris do not cover the entire domain of the selected observation times for one or '
            'more of your observation types. Values for those observations outside of the bounds of the ephemeris '
            'will be extrapolated. These values are not to be trusted.')
    if interpolation:
        warnings.warn(
            'Some of your observation times fall in the edges of the ephemeris domain, where interpolation does not'
            ' yield accurate results. The results are not to be trusted.')

    if 'A' in self.estimation_settings['estimated parameters']:
        if type(bodies.get('Phobos').rotation_model) is numerical_simulation.environment.TabulatedRotationalEphemeris:
            raise TypeError('Your estimated parameters include the libration amplitude. However, the ')

    return