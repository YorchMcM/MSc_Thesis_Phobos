import numpy as np
from os import getcwd
from Logistics import *

# These imports will go to both this files and all files importing this module
# Some of these imports will not be used in the present file, but will be in others.
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.numerical_simulation import environment, propagation
from tudatpy.util import result2array, compare_results
from tudatpy.kernel.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
from tudatpy.kernel.astro.element_conversion import quaternion_entries_to_rotation_matrix as quat2mat
from tudatpy.kernel.astro.element_conversion import cartesian_to_keplerian, true_to_mean_anomaly, semi_major_axis_to_mean_motion


def quaternion_entries_to_euler_angles(quaternion: np.ndarray[4]) -> np.ndarray[3]:

    rotation_matrix = quat2mat(quaternion)
    theta = np.arccos(rotation_matrix[2,2])
    psi = np.arctan2(rotation_matrix[0,2], -rotation_matrix[1,2])
    phi = np.arctan2(rotation_matrix[2,0], rotation_matrix[2,1])

    return np.array([psi, theta, phi])


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


def cartesian_to_keplerian_mean(state: np.ndarray[6], gravitational_parameter: float) -> np.ndarray[6]:

    keplerian = cartesian_to_keplerian(state, gravitational_parameter)
    e = keplerian[1]
    theta = keplerian[-1]
    mean_anomaly = true_to_mean_anomaly(e, theta)
    keplerian[-1] = mean_anomaly

    return keplerian


def cartesian_to_keplerian_mean_history(result: dict, gravitational_parameter: float) -> dict:

    keplerian_history = dict.fromkeys(list(result.keys()))
    for key in list(keplerian_history.keys()):
        keplerian_history[key] = cartesian_to_keplerian_mean(result[key], gravitational_parameter)

    return keplerian_history


def semi_major_axis_to_mean_motion_history(semi_major_axis_history: dict,
                                           gravitational_parameter: float) -> dict:

    mean_motion_history = dict.fromkeys(list(semi_major_axis_history.keys()))
    for key in list(mean_motion_history.keys()):
        mean_motion_history[key] = semi_major_axis_to_mean_motion(semi_major_axis_history[key],
                                                                  gravitational_parameter)

    return mean_motion_history


def normalize_spherical_harmonic_coefficients(cosine_coefficients: np.ndarray, sine_coefficients: np.ndarray) -> tuple:

    max_degree, max_order = cosine_coefficients.shape

    for degree in range(int(max_degree + 1)):
        for order in range(int(max_order + 1)):
            if order == 0 : delta = 1
            else : delta = 0
            N = np.sqrt((2 - delta)*(2*order+1)*np.math.factorial(order - degree)/np.math.factorial(order + degree))
            cosine_coefficients[degree, order] = cosine_coefficients[degree, order] / N  # Should this be a times or an over?
            sine_coefficients[degree, order] = sine_coefficients[degree, order] / N  # Should this be a times or an over?

    return cosine_coefficients, sine_coefficients


def get_martian_system(ephemerides: environment_setup.ephemeris.EphemerisSettings,
                       field_type: str,
                       field_source: str) -> environment.SystemOfBodies:

    '''
    This function will create the "bodies" object, just so we don't have all this lines in all models. Phobos is to be
    created from scratch. Its gravity field and ephemeris models have to be defined as part of the BodyListSettings, so
    they will have to be defined inside this function. These aspects are also model-dependent, so that they will have to
    be passed to the function as inputs.

    Main differences between models include ephemerides models, rotational models, and gravitational fields. The first
    and last are provided as inputs, so care is to be taken outside of this function. On the other hand, the rotational
    model is always set as synchronous. For model A1, the libration is added to the Body object itself, i.e. it is done
    outside of this function after calling it. For all other models (A2, B and C), the rotational model is irrelevant
    because Phobos' rotational dynamics are integrated. An inertia matrix is required for this, to be assigned to the
    Body object outside of this function.

    :param ephemerides: The ephemerides type to be assigned to Phobos.
    :param field_type: Either 'QUAD' or 'FULL'. The former creates the C20 and C22 coefficients. THe latter up to (4,4).
    :param field_source: Where to take the coefficients from. Two sources are possible: Le Maistre (2019) and Scheeres (2019).
    :return: bodies
    '''

    # WE FIRST CREATE MARS.
    bodies_to_create = ["Mars"]
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

    # AND LASTLY THE LIST OD BODIES IS CREATED.
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


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

    try:
        C_20 = gravity_field.cosine_coefficients[2,0]
        C_22 = gravity_field.cosine_coefficients[2,2]
    except:
        raise ValueError('Insufficient spherical harmonics for the computation of an inertia tensor.')

    R = gravity_field.reference_radius
    M = gravity_field.gravitational_parameter / constants.GRAVITATIONAL_CONSTANT

    N_20 = np.sqrt(5)
    N_22 = np.sqrt(10)

    C_20 = C_20 * N_20
    C_22 = C_22 * N_22

    aux = M*R**2
    I = (2/5)*aux
    A = aux * (C_20/3 - 2*C_22) + I
    B = aux * (C_20/3 + 2*C_22) + I
    C = aux * (-2*C_20/3) + I

    inertia_tensor = np.array([[A, 0, 0], [0, B, 0], [0, 0, C]])

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


def quaternion_to_euler_history(quaternion_history: dict) -> dict:

    euler_history = dict.fromkeys(list(quaternion_history.keys()))
    for key in list(euler_history.keys()):
        euler_history[key] = quaternion_entries_to_euler_angles(quaternion_history[key])

    return euler_history

def get_libration_history(translational_history: dict,
                          rotational_history: dict,
                          gravitational_parameter: float) -> dict:

    epochs_array = np.array(list(translational_history.keys()))
    keplerian_history = cartesian_to_keplerian_mean_history(translational_history,
                                                            gravitational_parameter)
    mean_anomaly_history = result2array(extract_element_from_history(keplerian_history, [-1]))
    quaternion_history = extract_element_from_history(rotational_history, [0, 1, 2, 3])
    euler_history = result2array(quaternion_to_euler_history(quaternion_history))[:,:4]

    libration_history = np.zeros([len(epochs_array), 2])
    libration_history[:,0] = epochs_array
    libration_history[:,1] = make_between_zero_and_twopi(euler_history[:,1] + euler_history[:,3] - mean_anomaly_history[:,1] - PI)

    return array2result(libration_history)
