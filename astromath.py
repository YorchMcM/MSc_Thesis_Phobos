from numpy.typing import ArrayLike
from numpy.fft import rfft, rfftfreq
from numpy.polynomial.polynomial import polyfit

from Logistics import *


def rms(array: np.ndarray) -> float:

    return np.sqrt((array @ array) / len(array) )


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


def euler_angles_to_rotation_matrix(euler_angles: ArrayLike) -> np.ndarray:

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
    :return: Rotated Euler angles

    '''

    R_AB = euler_angles_to_rotation_matrix(original_angles)
    R_CB = rotation_matrix.T @ R_AB
    new_euler_angles = rotation_matrix_to_313_euler_angles(R_CB)

    return new_euler_angles


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


def get_synodic_period(period1: float, period2: float) -> float:

    return 1.0/abs((1.0/period1)-(1.0/period2))


def get_position_rms_evolution(residual_history: dict, norm_position: bool) -> dict:

    residual_array = result2array(residual_history)
    if norm_position: number_of_iterations = int(residual_array.shape[1] - 2)
    else: number_of_iterations = int((residual_array.shape[1] - 1) / 3)
    iteration_list = list(range(number_of_iterations + 1))
    rms_evolution = dict.fromkeys(iteration_list)

    if norm_position:
        for idx in iteration_list: rms_evolution[idx] = rms(residual_array[:,idx+1])
    else:
        for idx in iteration_list:
            rms_evolution[idx] = np.zeros(3)
            for k in range(3):
                rms_evolution[idx][k] = rms(residual_array[:,3*(idx+1)+k+1])

    return rms_evolution


def extract_position_residuals(residual_history: np.ndarray,
                               observation_times: list[float],
                               norm_position: bool) -> dict:

    '''

    The old structure is as follows:

                        iter1       iter2       iter3       iter4       ...     ...     ...     interN
    x_first_epoch   :
    y_first_epoch   :
    z_first_epoch   :
    x_second_epoch  :
    y_second_epoch  :
    z_second_epoch  :
        ...
        ...
        ...
    x_last_epoch    :
    y_last_epoch    :
    z_last_epoch    :

    The new structure is going to be as follows for norm_position == False:


                      x_iter1, y_iter1, z_iter1, x_iter2, y_iter2, z_iter2, ... , x_iterN, y_iterN, z_iterN
    first_epoch     :
    second_epoch    :
    third_epoch     :
        ...
        ...
        ...
    last_epoch      :

    The new structure is going to be as follows for norm_position == False:


                      r_iter1, r_iter2, ... , r_iterN
    first_epoch     :
    second_epoch    :
    third_epoch     :
        ...
        ...
        ...
    last_epoch      :




    :param residual_history:
    :param observation_times:
    :param number_of_iterations:
    :param norm_position:
    :return:
    '''

    number_of_iterations = len(residual_history[0,:])
    if norm_position: N = number_of_iterations
    else: N = 3*number_of_iterations
    new_residual_history = dict.fromkeys(observation_times)
    for idx, epoch in enumerate(observation_times):
        current_array = np.zeros(N)
        for k in range(number_of_iterations):
            if norm_position:
                current_array[k] = np.linalg.norm(residual_history[3*idx:3*(idx+1),k])
            else:
                current_array[3*k:3*(k+1)] = residual_history[3*idx:3*(idx+1),k]
        new_residual_history[epoch] = current_array

    return new_residual_history


def covariance_to_correlation(covariance_matrix: np.ndarray) -> np.ndarray:

    correlation_matrix = np.zeros_like(covariance_matrix)

    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i,j] = covariance_matrix[i,j] / np.sqrt(covariance_matrix[i,i]*covariance_matrix[j,j])

    return correlation_matrix


def get_normalization_constant(degree: int, order: int) -> float:

    num = (2 - (order == 0))*(2*degree + 1)*np.math.factorial(degree - order)
    den = np.math.factorial(degree + order)
    N = np.sqrt( num / den )

    return N


def get_fourier_elements_from_history(result: dict,
                                      clean_signal: list = [0.0, 0]) -> tuple:
    '''
    The output of this function will be the frequencies in rad/unit of input, and the amplitudes.
    :param result:
    :param clean_signal: [jump_height for remove_jumps, degree of polynomial to fit]
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


def get_periapses(keplerian_history: dict) -> list:

    epochs_list = list(keplerian_history.keys())
    peri = [[None]*2]*len(epochs_list)

    true_anomaly = result2array(extract_elements_from_history(keplerian_history, [-1]))
    true_anomaly[:,1] = remove_jumps(true_anomaly[:,1], TWOPI)

    for idx in range(len(epochs_list[:-1])):
        if true_anomaly[idx,1] // TWOPI != true_anomaly[idx+1,1] // TWOPI:
            peri[idx] = [idx, true_anomaly[idx+1,0]]

    return [periapsis for periapsis in peri if periapsis != [None, None]]


def get_longitudinal_normal_mode_from_inertia_tensor(inertia_tensor: np.ndarray, mean_motion: float) -> float:

    # From Rambaux (2012) "Rotational motion of Phobos".

    A = inertia_tensor[0,0]
    B = inertia_tensor[1,1]
    C = inertia_tensor[2,2]
    gamma = (B - A) / C

    normal_mode = mean_motion * np.sqrt(3*gamma)

    return normal_mode