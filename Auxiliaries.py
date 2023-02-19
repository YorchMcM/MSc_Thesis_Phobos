import numpy as np
from os import getcwd

from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.kernel.numerical_simulation import environment, propagation
from tudatpy.kernel import constants
from tudatpy.util import result2array, compare_results


def norm_rows(array: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(array[k,:]) for k in range(array.shape[0])])


def norm_columns(array: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(array[:,k]) for k in range(array.shape[1])])


def str2vec(string: str, separator: str) -> np.ndarray:
    return np.array([float(element) for element in string.split(separator)])


def read_vector_history_from_file(file_name: str) -> dict:

    with open(file_name, 'r') as file: lines = file.readlines()
    keys = [float(line.split('\t')[0]) for line in lines]
    solution = dict.fromkeys(keys)
    for idx in range(len(keys)): solution[keys[idx]] = str2vec(lines[idx], '\t')[1:]

    return solution

def read_matrix_from_file(file_name: str, dimensions: list[int]) -> np.ndarray[float]:

    result = np.zeros(dimensions)
    with open(file_name, 'r') as file: rows = file.readlines()
    if len(rows) != dimensions[0]: raise ValueError('(read_matrix_from_file): Provided dimensions do not match with encountered matrix.')

    for idx1 in range(dimensions[0]):
        components = rows[idx1].split(' ')
        for idx2 in range(dimensions[1]):
            result[idx1, idx2] = float(components[idx2])

    return result


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


def let_there_be_a_gravitational_field(frame_name: str, field_type: str, source: str)\
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


def inertia_tensor_from_spherical_harmonic_gravity_field_settings(
        gravity_field_settings: environment_setup.gravity_field.SphericalHarmonicsGravityFieldSettings) -> np.ndarray:

    try:
        C_20 = gravity_field_settings.normalized_cosine_coefficients[2,0]
        C_22 = gravity_field_settings.normalized_cosine_coefficients[2,0]
    except:
        raise ValueError('Insufficient spherical harmonics for the computation of an inertia tensor.')

    R = gravity_field_settings.reference_radius
    M = gravity_field_settings.gravitational_parameter / constants.GRAVITATIONAL_CONSTANT

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
