import sys
import numpy as np
from os import getcwd
from Logistics import *
from numpy.fft import rfft, rfftfreq
from numpy.polynomial.polynomial import polyfit
import warnings

sys.path.insert(0, '/home/yorch/tudat-bundle/cmake-build-release/tudatpy')

# These imports will go to both this files and all files importing this module
# Some of these imports will not be used in the present file, but will be in others.
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.numerical_simulation import environment, propagation, estimation
from tudatpy.util import result2array, compare_results
from tudatpy.kernel.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
from tudatpy.kernel.astro.element_conversion import quaternion_entries_to_rotation_matrix as quat2mat
from tudatpy.kernel.astro.element_conversion import cartesian_to_keplerian, true_to_mean_anomaly, semi_major_axis_to_mean_motion
from tudatpy.plotting import trajectory_3d

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt


def rms(array: np.ndarray) -> float:

    return np.sqrt((array @ array) / len(array))


def rotation_matrix_x(angle: float) -> np.ndarray:

    return np.array([[1.0,            0.0,            0.0],
                     [0.0,  np.cos(angle),  np.sin(angle)],
                     [0.0, -np.sin(angle),  np.cos(angle)]])


def rotation_matrix_y(angle: float) -> np.ndarray:

    return np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                     [          0.0, 1.0,            0.0],
                     [np.sin(angle), 0.0, np.cos(angle)]])

def rotation_matrix_z(angle: float) -> np.ndarray:

    return np.array([[ np.cos(angle), np.sin(angle), 0.0],
                     [-np.sin(angle), np.cos(angle), 0.0],
                     [           0.0,           0.0, 1.0]])

def quaternion_entries_to_euler_angles(quaternion: np.ndarray[4]) -> np.ndarray[3]:

    rotation_matrix = quat2mat(quaternion)
    theta = np.arccos(rotation_matrix[2,2])
    psi = np.arctan2(rotation_matrix[0,2], -rotation_matrix[1,2])
    phi = np.arctan2(rotation_matrix[2,0], rotation_matrix[2,1])

    return np.array([psi, theta, phi])


def quaternion_to_euler_history(quaternion_history: dict) -> dict:

    epochs_list = list(quaternion_history.keys())
    euler_history = dict.fromkeys(epochs_list)
    for epoch in epochs_list:
        euler_history[epoch] = quaternion_entries_to_euler_angles(quaternion_history[epoch])

    return euler_history


def inertial2rsw(result: dict[float, np.ndarray], reference: dict[float, np.ndarray] = None) -> dict[float, np.ndarray]:

    if reference is not None and list(result.keys()) != list(reference.keys()):
        raise ValueError('(inertial2rsw): Incompatible inputs. Epoch keys do not match')

    result_rsw = dict.fromkeys(list(result.keys()))
    for key in list(result.keys()):
        if reference is not None: mat = inertial_to_rsw_rotation_matrix(reference[key])
        else: mat = inertial_to_rsw_rotation_matrix(result[key])
        pos = mat @ result[key][:3]
        vel = mat @ result[key][3:]
        result_rsw[key] = np.array([pos, vel]).reshape(6)

    return result_rsw


def cartesian_to_keplerian_history(cartesian_history: dict,
                                   gravitational_parameter: float) -> dict:

    epochs_list = list(cartesian_history.keys())
    keplerian_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        keplerian_history[key] = cartesian_to_keplerian(cartesian_history[key], gravitational_parameter)

    return keplerian_history


def mean_motion_history_from_keplerian_history(keplerian_history: dict,
                                               gravitational_parameter: float) -> dict:

    epochs_list = list(keplerian_history.keys())
    mean_motion_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        mean_motion_history[key] = semi_major_axis_to_mean_motion(keplerian_history[key][0],
                                                                  gravitational_parameter)

    return mean_motion_history


def mean_anomaly_history_from_keplerian_history(keplerian_history: dict) -> dict:

    epochs_list = list(keplerian_history.keys())
    mean_anomaly_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        eccentricity = keplerian_history[key][1]
        true_anomaly = keplerian_history[key][-1]
        mean_anomaly_history[key] = true_to_mean_anomaly(eccentricity, true_anomaly)

    return mean_anomaly_history


def mean_anomaly_history_from_cartesian_history(cartesian_history: dict,
                                                gravitational_parameter: float) -> dict:

    keplerian_history = cartesian_to_keplerian_history(cartesian_history, gravitational_parameter)
    mean_anomaly_history = mean_anomaly_history_from_keplerian_history(keplerian_history)

    return mean_anomaly_history


def get_normalization_constant(degree: int, order: int) -> float:

    num = (2 - (order == 0))*(2*degree + 1)*np.math.factorial(degree - order)
    den = np.math.factorial(degree + order)
    N = np.sqrt( num / den )

    return N


def get_solar_system(ephemerides: environment_setup.ephemeris.EphemerisSettings,
                     field_type: str,
                     field_source: str,
                     scaled_amplitude: float = 0.0) -> environment.SystemOfBodies:

    '''
    This function will create the "bodies" object, just so we don't have all this lines in all models. Phobos is to be
    created from scratch. Its gravity field and ephemeris models have to be defined as part of the BodyListSettings, so
    they will have to be defined inside this function. These aspects are also model-dependent, so that they will have to
    be passed to the function as inputs.

    Main differences between models include ephemerides models, rotational models, and gravitational fields. The first
    and last are provided as inputs, so care is to be taken outside of this function. On the other hand, the rotational
    model is always set as synchronous. For model A1, the libration is added to the Body object itself, after creating
    the ListOfBodies object, but still inside this function. For all other models (A2, B and C), the rotational model is
    irrelevant because Phobos' rotational dynamics are integrated. An inertia matrix is required for this, and is also
    assigned inside this function. Note that this attribute of Phobos will not be used in model A1.

    The result is a Phobos with the specified gravity field, a rotation model, and an inertia tensor.

    :param ephemerides: The ephemerides type to be assigned to Phobos.
    :param field_type: Either 'QUAD' or 'FULL'. The former creates the C20 and C22 coefficients. THe latter up to (4,4).
    :param field_source: Where to take the coefficients from. Two sources are possible: Le Maistre (2019) and Scheeres (2019).
    :param scaled_amplitude: The scaled libration amplitude. Note that the rotation model will ONLY be used in model A1.
    :return: bodies
    '''

    # WE FIRST CREATE MARS.
    bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Deimos", "Jupiter", "Saturn"]
    global_frame_origin = "Mars"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

    # WE THEN CREATE PHOBOS USING THE INPUTS.
    body_settings.add_empty_settings('Phobos')
    # Ephemeris and rotation models.
    body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
    body_settings.get('Phobos').ephemeris_settings = ephemerides
    # Gravity field.
    body_settings.get('Phobos').gravity_field_settings = get_gravitational_field('Phobos_body_fixed', field_type, field_source)

    # AND LASTLY THE LIST OF BODIES IS CREATED.
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # There are some properties that are not assigned to Phobos' body settings, but rather to the body object itself.
    # One is the rotation model (only used in model A1).
    bodies.get('Phobos').rotation_model.libration_calculator = environment.DirectLongitudeLibrationCalculator(scaled_amplitude)
    # Another is the inertia tensor (useless in model A1).
    bodies.get('Phobos').inertia_tensor = inertia_tensor_from_spherical_harmonic_gravity_field(
        bodies.get('Phobos').gravity_field_model
    )

    return bodies


def get_model_a1_propagator_settings(bodies: environment.SystemOfBodies,
                                     simulation_time: float,
                                     initial_epoch: float = 0.0,
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
    time_step = 450.0  # These are 300s = 5min / 450s = 7.5min
    coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                      coefficients,
                                                                                      time_step,
                                                                                      time_step,
                                                                                      np.inf, np.inf)
    # PROPAGATION SETTINGS
    # Initial conditions
    initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'NONE', initial_epoch)
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


def get_model_a2_propagator_settings(bodies: environment.SystemOfBodies,
                                     simulation_time: float,
                                     initial_epoch: float,
                                     initial_state: np.ndarray,
                                     dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [])\
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
                                     # Moon=[torque_on_phobos],
                                     Deimos=[torque_on_phobos],
                                     Jupiter=[torque_on_phobos]
                                     # Saturn=[torque_on_phobos]
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
    # Initial conditions
    # initial_epoch = 13035.0  # This is (approximately) the first perisapsis passage since J2000
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

    #check_ephemeris_sufficiency(bodies, initial_epoch + simulation_time)
    return propagator_settings


def get_model_b_propagator_settings(bodies: environment.SystemOfBodies,
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
    # Initial conditions
    # initial_epoch = 13035.0  # This is (aproximately) the first perisapsis passage since J2000
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

    #check_ephemeris_sufficiency(bodies, initial_epoch + simulation_time)
    return propagator_settings


def get_fake_initial_state(bodies: environment.SystemOfBodies,
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


def get_gravitational_field(frame_name: str, field_type: str, source: str)\
        -> environment_setup.gravity_field.GravityFieldSettings:

    # I set the frame_name to be an input because this has to be consistent with other parts of the program, so that it
    # is easy to check that from the main script without coming here, and can be easily changed if necessary.
    # Furthermore, different models will require different types of gravity fields. That is also an input.

    datadir = getcwd() + '/Normalized gravity fields/'
    if source == 'Scheeres':
        phobos_gravitational_parameter = 713000.0
        phobos_reference_radius = 11118.81652
    elif source == 'Le Maistre':
        phobos_gravitational_parameter = 1.06e16*constants.GRAVITATIONAL_CONSTANT
        phobos_reference_radius = 14e3
    else: raise ValueError('(let_there_be_a_gravitational_field): Invalid source. Only "Scheeres" and "Le Maistre are '
                           'allowed. You provided "' + source + '".')

    cosines_file = datadir + 'cosines ' + source + '.txt'
    sines_file = datadir + 'sines ' + source + '.txt'
    phobos_normalized_cosine_coefficients = read_matrix_from_file(cosines_file, [5,5])
    phobos_normalized_sine_coefficients = read_matrix_from_file(sines_file, [5, 5])

    if field_type not in ['QUAD', 'FULL']:
        raise ValueError('(let_there_be_a_gravitational_field): Wrong field type. Only "FULL" and "QUAD" are supported. "'
                         + field_type + '" was provided.')

    if field_type == 'QUAD':

        c20 = phobos_normalized_cosine_coefficients[2,0]
        c22 = phobos_normalized_cosine_coefficients[2,2]
        phobos_normalized_cosine_coefficients = np.zeros_like(phobos_normalized_cosine_coefficients)
        phobos_normalized_cosine_coefficients[0,0] = 1.0
        phobos_normalized_cosine_coefficients[2,0] = c20
        phobos_normalized_cosine_coefficients[2,2] = c22
        phobos_normalized_sine_coefficients = np.zeros_like(phobos_normalized_sine_coefficients)

    settings_to_return = environment_setup.gravity_field.spherical_harmonic(
        phobos_gravitational_parameter,
        phobos_reference_radius,
        phobos_normalized_cosine_coefficients,
        phobos_normalized_sine_coefficients,
        associated_reference_frame = frame_name)

    return settings_to_return


def inertia_tensor_from_spherical_harmonic_gravity_field(
        gravity_field: environment.SphericalHarmonicsGravityField) -> np.ndarray:

    '''
    This function is completely equivalent to the tudat-provided getInertiaTensor function defined in the file
    tudat.astro.gravitation.sphericalHarmonicsGravityField.cpp, while requiring less inputs.
    NOTE: Phobos' mean moment
    of inertia is hard-coded inside this function.
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


def get_benchmark_integrator_settings(step_size: float) -> tuple:

    '''
    The benchmark simulation will be performed using a RKDP8(7) integrator. Different time steps will be studied. The
    benchmark error will be computed as the difference with another simulation with half the step size. Thus, for each
    time step Δt, two simulations will be run: one with time step Δt and one with time step 0.5Δt.
    '''

    base_benchmark_time_step = step_size / 2.0
    top_benchmark_time_step = step_size

    rkdp_coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    base_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(base_benchmark_time_step,
                                                                                           rkdp_coefficients,
                                                                                           base_benchmark_time_step,
                                                                                           base_benchmark_time_step,
                                                                                           np.inf, np.inf)
    top_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(top_benchmark_time_step,
                                                                                          rkdp_coefficients,
                                                                                          top_benchmark_time_step,
                                                                                          top_benchmark_time_step,
                                                                                          np.inf, np.inf)

    return base_integrator_settings, top_integrator_settings


def quaternion_to_matrix_history(quaternion_history: dict) -> dict:

    epochs_list = list(quaternion_history.keys())
    rotation_matrix_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        rotation_matrix_history[key] = quat2mat(quaternion_history[key])

    return rotation_matrix_history


def get_libration_history(translational_history: dict,
                          rotational_history: dict) -> dict:

    epochs_list = list(rotational_history.keys())
    quaternion_history = extract_elements_from_history(rotational_history, [0, 1, 2, 3])
    rotation_matrix_history = quaternion_to_matrix_history(quaternion_history)

    libration_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        rtn_to_inertial_matrix = inertial_to_rsw_rotation_matrix(translational_history[key]).T
        r = rtn_to_inertial_matrix[:,0]
        t = rtn_to_inertial_matrix[:,1]
        x = rotation_matrix_history[key][:,0]
        cosine_libration = np.dot(x, -r)
        sine_libration = np.dot(x, t)
        libration_history[key] = np.arctan2(sine_libration, cosine_libration)

    return libration_history


def get_longitudinal_libration_history_from_libration_calculator(translational_history: dict,
                                                                 gravitational_parameter: float,
                                                                 libration_amplitude: float) -> dict:
    epochs_list = list(translational_history.keys())
    keplerian_history = cartesian_to_keplerian_history(translational_history, gravitational_parameter)
    e = extract_elements_from_history(keplerian_history, 1)
    libration_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        r = translational_history[key][:3]
        v = translational_history[key][3:]
        libration_history[key] = np.dot(r, v) / np.linalg.norm(np.cross(r, v))
        libration_history[key] = libration_history[key] * np.sqrt(1 - e[key] * e[key]) / e[key]
        libration_history[key] = libration_history[key] * libration_amplitude

    return libration_history


def get_fourier_elements_from_history(result: dict,
                                      clean_signal: list = [0.0, 0]) -> tuple:
    '''
    The output of this function will be the frequencies in rad/unit of input, and the amplitudes.
    :param result:
    :param clean_signal:
    :return:
    '''

    result_array = result2array(result)
    sample_times = result_array[:,0]
    signal = result_array[:,1]

    if len(sample_times) % 2.0 != 0.0:
        sample_times = sample_times[:-1]
        signal = signal[:-1]

    if clean_signal[0] != 0.0: signal = remove_jumps(signal, clean_signal[0])
    if clean_signal[1] != 0:
        coeffs = polyfit(sample_times, signal, clean_signal[1])
        signal = signal - coeffs[0] - coeffs[1] * sample_times

    n = len(sample_times)
    dt = sample_times[1] - sample_times[0]
    frequencies = TWOPI * rfftfreq(n, dt)
    amplitudes = 2*abs(rfft(signal, norm = 'forward'))

    return frequencies, amplitudes

def plot_kepler_elements(keplerian_history: dict, title: str = None) -> None:

    epochs_array = np.array(list(keplerian_history.keys()))

    (fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))) = plt.subplots(3, 2)
    # Semi-major axis
    a = result2array(extract_elements_from_history(keplerian_history, 0))[:, 1]
    ax1.plot(epochs_array / 86400.0, a / 1000.0)
    ax1.set_xlabel('Time [days since J2000]')
    ax1.set_ylabel(r'$a$ [km]')
    ax1.set_title('Semimajor axis')
    ax1.grid()
    # Eccentricity
    e = result2array(extract_elements_from_history(keplerian_history, 1))[:, 1]
    ax2.plot(epochs_array / 86400.0, e)
    ax2.set_xlabel('Time [days since J2000]')
    ax2.set_ylabel(r'$e$ [-]')
    ax2.set_title('Eccentricity')
    ax2.grid()
    # Inclination
    i = result2array(extract_elements_from_history(keplerian_history, 2))[:, 1]
    ax3.plot(epochs_array / 86400.0, i * 360.0 / TWOPI)
    ax3.set_xlabel('Time [days since J2000]')
    ax3.set_ylabel(r'$i$ [º]')
    ax3.set_title('Inclination')
    ax3.grid()
    # Righ-ascension of ascending node
    RAAN = result2array(extract_elements_from_history(keplerian_history, 4))[:, 1]
    ax4.plot(epochs_array / 86400.0, RAAN * 360.0 / TWOPI)
    ax4.set_xlabel('Time [days since J2000]')
    ax4.set_ylabel(r'$\Omega$ [º]')
    ax4.set_title('RAAN')
    ax4.grid()
    # Argument of periapsis
    omega = result2array(extract_elements_from_history(keplerian_history, 3))[:, 1]
    ax5.plot(epochs_array / 86400.0, omega * 360.0 / TWOPI)
    ax5.set_xlabel('Time [days since J2000]')
    ax5.set_ylabel(r'$\omega$ [º]')
    ax5.set_title('Argument of periapsis')
    ax5.grid()
    # True anomaly
    theta = result2array(extract_elements_from_history(keplerian_history, 5))[:, 1]
    ax6.plot(epochs_array / 86400.0, theta * 360.0 / TWOPI)
    ax6.set_xlabel('Time [days since J2000]')
    ax6.set_ylabel(r'$\theta$ [º]')
    ax6.set_title('True anomaly')
    ax6.grid()

    fig.tight_layout()
    if title is None: fig.suptitle('Keplerian elements')
    else: fig.suptitle(title)

    return


def bring_history_inside_bounds(original: dict, lower_bound: float,
                                upper_bound: float, include: str = 'lower') -> np.ndarray:

    original_array = result2array(original)
    original_array[:,1:] = bring_inside_bounds(original_array[:,1:], lower_bound, upper_bound, include)
    new = array2result(original_array)

    return new


def get_longitudinal_normal_mode_from_inertia_tensor(inertia_tensor: np.ndarray, mean_motion: float) -> float:

    # From Rambaux (2012) "Rotational motion of Phobos".

    A = inertia_tensor[0,0]
    B = inertia_tensor[1,1]
    C = inertia_tensor[2,2]
    gamma = (B - A) / C

    normal_mode = mean_motion * np.sqrt(3*gamma)

    return normal_mode


def acceleration_norm_from_body_on_phobos(body_exerting_acceleration: str)\
        -> propagation_setup.dependent_variable.SingleDependentVariableSaveSettings:

    point_mass = propagation_setup.acceleration.point_mass_gravity_type
    ret = propagation_setup.dependent_variable.single_acceleration_norm(point_mass, 'Phobos', body_exerting_acceleration)

    return ret


def torque_norm_from_body_on_phobos(body_exerting_torque: str) \
        -> propagation_setup.dependent_variable.SingleDependentVariableSaveSettings:

    point_mass = propagation_setup.torque.spherical_harmonic_gravitational_type
    ret = propagation_setup.dependent_variable.single_torque_norm(point_mass, 'Phobos', body_exerting_torque)

    return ret


def get_periapses(keplerian_history: dict) -> list:

    epochs_list = list(keplerian_history.keys())
    peri = [[None]*2]*len(epochs_list)

    true_anomaly = result2array(extract_elements_from_history(keplerian_history, [-1]))
    true_anomaly[:,1] = remove_jumps(true_anomaly[:,1], TWOPI)

    for idx in range(len(epochs_list[:-1])):
        if true_anomaly[idx,1] // TWOPI != true_anomaly[idx+1,1] // TWOPI:
            peri[idx] = [idx, true_anomaly[idx+1,0]]

    return [periapsis for periapsis in peri if periapsis != [None, None]]


def average_mean_motion_over_integer_number_of_orbits(keplerian_history: dict, gravitational_parameter: float) -> float:

    mean_motion_history = mean_motion_history_from_keplerian_history(keplerian_history, gravitational_parameter)
    periapses = get_periapses(keplerian_history)
    first_periapsis = periapses[0][0]
    last_periapsis = periapses[-1][0]
    mean_motion_over_integer_number_of_orbits = np.array(list(mean_motion_history.values())[first_periapsis:last_periapsis])

    return np.mean(mean_motion_over_integer_number_of_orbits), len(periapses)


def get_synodic_period(period1: float, period2: float) -> float:

    return 1.0/abs((1.0/period1)-(1.0/period2))


def check_ephemeris_sufficiency(bodies: environment.SystemOfBodies, max_simulation_epoch: float) -> None:

    available_states = bodies.get('Phobos').ephemeris.body_state_history
    max_ephemeris_epoch = list(available_states.keys())[-1]

    if max_ephemeris_epoch < max_simulation_epoch:
        raise Warning('Insufficient ephemerides loaded for requested propagation.\nLargest simulation epoch: '
                      + str(max_simulation_epoch) + '\nMax ephemeris epoch: ' + str(max_ephemeris_epoch))

    return


def rotation_matrix_to_313_euler_angles(matrix: np.ndarray) -> np.ndarray:

    '''

    This function is a direct implementation of Eq.(A6) in Fukushima (2012). Given a matrix R such that v = Ru, where u
    is a vector expressed in body-fixed frame and v is expressed in inertial frame, it returns the 3-1-3 Euler angles
    that define this rotation.

    :param matrix: Rotation matrix.
    :return: 3-1-3 Euler angles

    '''

    psi = bring_inside_bounds(np.arctan2(matrix[0,2], -matrix[1,2]), 0.0, TWOPI)
    theta = np.arccos(matrix[2,2])
    phi = bring_inside_bounds(np.arctan2(matrix[2,0], matrix[2,1]), 0.0, TWOPI)

    return np.array([psi, theta, phi])


def euler_angles_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:

    '''

    Given the three Euler angles that characterize a rotation, this function returns a matrix R such that v = Ru,
    where u is a vector expressed in body-fixed frame and v is expressed in inertial frame.

    :param euler_angles: 3-1-3 Euler angles
    :return: Rotation matrix

    '''

    psi, theta, phi = euler_angles

    R_psi = rotation_matrix_z(psi)
    R_theta = rotation_matrix_x(theta)
    R_phi = rotation_matrix_z(phi)

    body_fixed_to_inertial = R_phi @ R_theta @ R_psi  # This is the TRANSPOSE of Eq.(A4) in Fukushima et al. (2012)
    body_fixed_to_inertial = body_fixed_to_inertial.T

    return body_fixed_to_inertial


def rotate_euler_angles(original_angles: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:

    '''

    Consider three reference frames A, B and C. Consider the three angles that define the rotation from frame A to frame
    B, and the associated rotation matrix R^(A/B), such that u^(A) = R^(A/B)u^(B). We now want the rotation matrix
    R^(C/B) such that u^(C) = R^(C/B)u^(B) and the three Euler angles associated to that rotation. This function returns
    these three Euler angles.

    For this, one needs the original three Euler angles and a definition for the rotation between frames A and C. This
    function will take the rotation matrix R^(A/C), such that u^(A) = R^(A/C)u^(C).

    Given these inputs, the first step will be to obtain R^(A/B), then obtain R^(C/B) = R^(C/A)*R^(A/B), where R^(C/A)
    is the inverse and transpose of R^(A/C), and then extract the three Euler angles of this new R^(C/B).

    :param original_euler_angles: Original 3-1-3 Euler angles
    :param rotation_matrix: R^(A/C)
    :return: Rotation matrix

    '''

    R_AB = euler_angles_to_rotation_matrix(original_angles)
    R_CB = rotation_matrix.T @ R_AB
    new_euler_angles = rotation_matrix_to_313_euler_angles(R_CB)

    return new_euler_angles


class MarsEquatorOfDate():

    def __init__(self, bodies: environment.SystemOfBodies):

        self.alpha_0 = np.radians(317.269202)
        self.delta_0 = np.radians(54.432516)
        self.W = np.radians(176.049863)

        self.phobos = bodies.get('Phobos')

        self.mars_to_j2000_rotation = self.get_mars_to_J2000_rotation_matrix()
        self.j2000_to_mars_rotation = self.mars_to_j2000_rotation.T

        return

    def get_mars_to_J2000_rotation_matrix(self):

        psi = bring_inside_bounds(PI/2 + self.alpha_0, 0.0, TWOPI)
        theta = bring_inside_bounds(PI/2 - self.delta_0, 0.0, TWOPI)
        phi = bring_inside_bounds(self.W, 0.0, TWOPI)

        return euler_angles_to_rotation_matrix([psi, theta, phi])

    def get_euler_angles_wrt_mars_equator(self) -> np.ndarray:

        phobos_to_J2000_rotation_matrix = self.phobos.body_fixed_to_inertial_frame
        phobos_to_mars_rotation_matrix = self.j2000_to_mars_rotation @ phobos_to_J2000_rotation_matrix

        return rotation_matrix_to_313_euler_angles(phobos_to_mars_rotation_matrix)

    def rotate_euler_angles_from_J2000_to_mars_equator(self, euler_angles_j2000: np.ndarray) -> np.ndarray:

        return rotate_euler_angles(euler_angles_j2000, self.mars_j2000_rotation)


def get_true_initial_state(model: str, initial_epoch: float) -> np.ndarray:

    if model == 'B': ephemeris_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
    if model == 'C': ephemeris_file = '/home/yorch/thesis/everything-works-results/model-c/states-d8192.txt'

    state_history = read_vector_history_from_file(ephemeris_file)
    initial_state = state_history[initial_epoch]

    return initial_state


def extract_estimation_output(estimation_output: estimation.EstimationOutput,
                              observation_times: list[float],
                              residual_type: str) -> tuple:

    if residual_type not in ['position', 'orientation']:
        raise ValueError('(): Invalid residual type. Only "position" and "orientation" are allowed. Residual type provided is "' + residual_type + '".')

    number_of_iterations = estimation_output.residual_history.shape[1]
    iteration_array = list(range(number_of_iterations + 1))
    if residual_type == 'position':
        residual_history = extract_position_residuals(estimation_output.residual_history,
                                                      observation_times,
                                                      number_of_iterations)
        residual_rms_evolution = get_position_rms_evolution(residual_history)
    if residual_type == 'orientation':
        residual_history = extract_orientation_residuals(estimation_output.residual_history,
                                                         observation_times,
                                                         number_of_iterations)
        residual_rms_evolution = get_orientation_rms_evolution(residual_history)
    parameter_evolution = dict(zip(iteration_array, estimation_output.parameter_history.T))




    return residual_history, parameter_evolution, residual_rms_evolution


def extract_position_residuals(residual_history: np.ndarray, observation_times: list[float], number_of_iterations: float) -> dict:

    '''

    The new structure is going to be as follows:

    first_epoch     : x_iter1, y_iter1, z_iter1, x_iter2, y_iter2, z_iter2, ... , x_iterN, y_iterN, z_iterN
    second_epoch    : x_iter1, y_iter1, z_iter1, x_iter2, y_iter2, z_iter2, ... , x_iterN, y_iterN, z_iterN
        ...
        ...
        ...
    last_epoch      : x_iter1, y_iter1, z_iter1, x_iter2, y_iter2, z_iter2, ... , x_iterN, y_iterN, z_iterN


    :param residual_history:
    :param observation_times:
    :param number_of_iterations:
    :return:
    '''

    new_residual_history = dict.fromkeys(observation_times)
    for idx, epoch in enumerate(observation_times):
        current_array = np.zeros(3 * number_of_iterations)
        for k in range(number_of_iterations):
            current_array[3*k:3*(k+1)] = residual_history[3*idx:3*(idx+1),k]
        new_residual_history[epoch] = current_array

    return new_residual_history


def extract_orientation_residuals(residual_history: np.ndarray, observation_times: np.ndarray, number_of_iterations: float) -> dict:

    return


def get_position_rms_evolution(residual_history: dict) -> dict:

    residual_array = result2array(residual_history)
    number_of_iterations = int((residual_array.shape[1] - 1) / 3)
    iteration_list = list(range(1, number_of_iterations + 1))
    rms_evolution = dict.fromkeys(iteration_list)

    for idx in iteration_list:
        rms_evolution[idx] = np.zeros(3)
        for k in range(3):
            rms_evolution[idx][k] = rms(residual_array[:,3*(idx-1)+k+1])

    return rms_evolution


def get_orientation_rms_evolution(residual_history: dict) -> dict:

    return