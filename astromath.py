from numpy.typing import ArrayLike
from numpy.fft import rfft, rfftfreq
from numpy.polynomial.polynomial import polyfit
from numpy.linalg import norm

from Logistics import *

'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                       GENERIC MATH                                                 ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


def norm_columns(array: np.ndarray) -> np.ndarray:
    return norm(array, axis = 0)


def norm_history(history: dict) -> dict:

    epochs = list(history.keys())
    norms = np.vstack([np.linalg.norm(value) for value in list(history.values())])

    return dict(zip(epochs, norms))


def norm_rows(array: np.ndarray) -> np.ndarray:
    return norm(array, axis = 1)


def rms(array: np.ndarray) -> float:
    return np.sqrt((array @ array) / len(array) )


def unit(vector: np.ndarray) -> np.ndarray:

    temp = norm(vector)
    if temp == 0.0:
        to_return = vector
    else:
        to_return = vector / temp

    return to_return


'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                           ROTATIONS                                                ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


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


'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                    FREQUENCY ANALYSIS                                              ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


def fourier_transform(time_history: np.ndarray | dict, clean_signal: list = [0.0, 0]) -> np.ndarray:

    """

    This function computes the fast fourier transform of a provided time history. It assumes that the quantity of the time history is real, and calls Numpy's rfft function to compute it. This function complements Numpy's rfft in the following ways:

    · It accounts for a time history with an odd number of entries and removes the last entry to make it of even length.
    · It allows to clean the signal. This encompasses two things:
        - Some quantities present jumps because they are by definition bounded inside an interval, but their evolution is secular. This function removes this jumps and works with a continuous signal.
        - Sometimes one is interested in the residuals of the signal when a predefined polynomial is removed from it. This function allows to remove this polynomial and return the fft of the residuals. The coefficients of the polynomial are computed using Numpy's polyfit.
    · Numpy's rfft returns a complex arrays of coefficients, usually not useful. This function returns the amplitude domain, attending to the fact that (a) the norm of the coefficients is to be taken and (b) the actual amplitude of the sinusoid is twice the norm of the complex coefficient.
    · Numpy's rfftfreq returns a frequency array that is in cycles / unit_of_time. This function returns the frequencies in rad / unit_of_time.

    Parameters
    ----------
    time_history: np.ndarray
        A two-dimensional array with two columns: the first column is the time, the second is the quantity whose frequency content is to be computed.
    clean_signal: list[float]
        This determines (a) whether the signal is to be removed of jumps and (b) whether a polynomial is to be removed from the signal. The first entry of clean_signal is the value of the jumps, and the second entry is the degree of the polynomial.

    Returns
    -------
    np.ndarray
        An array with two columns: the first column contains the frequencies (in rad / unit_of_time), the second contains the amplitudes.

    """

    if type(clean_signal[1]) != int:
        raise TypeError('(fourier_transform): Invalid input. The second entry in clean_signal should be of type "int". A type ' + str(type(clean_signal[1])) + 'was provided.')
    if clean_signal[1] < 0:
        raise ValueError('(fourier_transform): Invalid input. The second entry in clean_signal cannot be negative. Current values is ' + str(clean_signal[1]) + '.')
    if clean_signal[0] < 0.0:
        raise ValueError('(fourier_transform): Invalid input. The first entry in clean_signal cannot be negative. Current values is ' + str(clean_signal[1]) + '.')

    if type(time_history) == dict:
        time_history = dict2array(time_history)

    sample_times = time_history[:,0]
    signal = time_history[:,1:]

    if len(sample_times) % 2.0 != 0.0:
        sample_times = sample_times[:-1]
        signal = signal[:-1]

    if clean_signal[0] != 0.0:
        signal = remove_jumps(signal, clean_signal[0])
    if clean_signal[1] != 0:
        coeffs = polyfit(sample_times, signal, clean_signal[1])
        for idx, current_coeff in enumerate(coeffs):
            exponent = idx
            signal = signal - np.atleast_2d(sample_times**exponent).T @ np.atleast_2d(current_coeff)

    n = len(sample_times)
    dt = sample_times[1] - sample_times[0]
    frequencies = TWOPI * rfftfreq(n, dt)
    amplitudes = 2*abs(rfft(signal, axis = 0, norm = 'forward'))

    if len(amplitudes.shape) == 1:
        return np.concatenate((np.atleast_2d(frequencies).T, np.atleast_2d(amplitudes).T), axis = 1)
    else:
        return np.concatenate((np.atleast_2d(frequencies).T, amplitudes), axis=1)

'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                              ASTRO / PHYSICAL PROPERTIES AND QUANTITIES                            ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


def average_over_integer_number_of_orbits(quantity_history: np.ndarray, reference_keplerian_history: np.ndarray) -> tuple:

    periapses = get_periapses(reference_keplerian_history)
    first_periapsis = periapses[0][0]
    last_periapsis = periapses[-1][0]
    quantity_over_integer_number_of_orbits = quantity_history[first_periapsis:last_periapsis,1:]

    return np.mean(quantity_over_integer_number_of_orbits, axis = 0), len(periapses)


def compute_eccentricity_from_dependent_variables(dependents: dict) -> float:

    dependents_array = dict2array(dependents)
    eccentricity_history = dependents_array[:,[0,8]]
    keplerian_history = dependents_array[:,[0, 7, 8, 9, 10, 11, 12]]
    average_eccentricity, trash = average_over_integer_number_of_orbits(eccentricity_history, keplerian_history)

    return average_eccentricity


def compute_scaled_libration_amplitude_from_dependent_variables(dependents: dict) -> float:

    longitude = dict2array(dependents)[:, [0, 6]]
    freq_amp = fourier_transform(longitude)
    tidal_libration_amplitude = max(freq_amp[:,1])

    return tidal_libration_amplitude / compute_eccentricity_from_dependent_variables(dependents)


def get_longitudinal_normal_mode_from_inertia_tensor(inertia_tensor: np.ndarray, mean_motion: float) -> float:

    # From Rambaux (2012) "Rotational motion of Phobos".

    A = inertia_tensor[0,0]
    B = inertia_tensor[1,1]
    C = inertia_tensor[2,2]
    gamma = (B - A) / C

    normal_mode = mean_motion * np.sqrt(3*gamma)

    return normal_mode


def get_normalization_constant(degree: int, order: int) -> float:

    num = (2 - (order == 0))*(2*degree + 1)*np.math.factorial(degree - order)
    den = np.math.factorial(degree + order)
    N = np.sqrt( num / den )

    return N


def get_periapses(keplerian_history: np.ndarray) -> list:

    epochs_list = keplerian_history[:,0]
    peri = [[None]*2]*len(epochs_list)

    true_anomaly = keplerian_history[:,[0,-1]]
    true_anomaly[:,1] = remove_jumps(true_anomaly[:,1], TWOPI)

    for idx in range(len(epochs_list[:-1])):
        if true_anomaly[idx,1] // TWOPI != true_anomaly[idx+1,1] // TWOPI:
            peri[idx] = [idx, true_anomaly[idx+1,0]]

    return [periapsis for periapsis in peri if periapsis != [None, None]]


def get_synodic_period(period1: float, period2: float) -> float:
    return 1.0 / abs( (1.0/period1)-(1.0/period2) )


'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                              ESTIMATION REARRANGEMENT OPERATIONS                                   ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


def get_position_statistical_indicators_evolution(residual_histories: list[dict],
                                                  norm_residuals: bool,
                                                  convert_residuals_to_rsw: bool) -> dict:

    residual_array = dict2array(residual_histories[0])
    number_of_iterations = int((residual_array.shape[1] - 4) / 3)
    iteration_list = list(range(number_of_iterations + 1))

    indicators_evolution = [dict.fromkeys(iteration_list)]

    for idx in iteration_list:
        indicators_evolution[0][idx] = np.zeros(12)
        for k in range(3):
            indicators_evolution[0][idx][4*k:4*(k+1)] = np.array([min(residual_array[:,3*idx+k+1]),
                                                                  np.mean(residual_array[:,3*idx+k+1]),
                                                                  rms(residual_array[:,3*idx+k+1]),
                                                                  max(residual_array[:,3*idx+k+1])])

    if norm_residuals:
        indicators_evolution.append(dict.fromkeys(iteration_list))
        residual_array = dict2array(residual_histories[1])
        for idx in iteration_list:
            indicators_evolution[1][idx] = np.array([min(residual_array[:,idx+1]),
                                                     np.mean(residual_array[:,idx+1]),
                                                     rms(residual_array[:,idx+1]),
                                                     max(residual_array[:,idx+1])])

    if convert_residuals_to_rsw:
        indicators_evolution.append(dict.fromkeys(iteration_list))
        residual_array = dict2array(residual_histories[-1])
        for idx in iteration_list:
            indicators_evolution[-1][idx] = np.zeros(12)
            for k in range(3):
                indicators_evolution[-1][idx][4*k:4*(k+1)] = np.array([min(residual_array[:,3*idx+k+1]),
                                                                       np.mean(residual_array[:,3*idx+k+1]),
                                                                       rms(residual_array[:,3*idx+k+1]),
                                                                       max(residual_array[:,3*idx+k+1])])

    return indicators_evolution


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

    The new structure is going to be as follows for norm_position == True:


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


def rearrange_position_residuals(residual_history: np.ndarray,
                                 ephemeris_states_at_observation_times: dict[float, np.ndarray]) -> dict:

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

    The new structure is going to be as follows:


                      x_iter1, y_iter1, z_iter1, x_iter2, y_iter2, z_iter2, ... , x_iterN, y_iterN, z_iterN
    first_epoch     :
    second_epoch    :
    third_epoch     :
        ...
        ...
        ...
    last_epoch      :

    :param residual_history:
    :param ephemeris_states_at_observation_times:
    :return:

    '''

    observation_times = list(ephemeris_states_at_observation_times.keys())
    number_of_iterations = len(residual_history[0, :])
    N = 3 * number_of_iterations
    new_residual_history = dict.fromkeys(observation_times)
    for idx, epoch in enumerate(observation_times):
        current_array = np.zeros(N)
        for k in range(number_of_iterations):
            current_array[3 * k:3 * (k + 1)] = residual_history[3 * idx:3 * (idx + 1), k]
        new_residual_history[epoch] = current_array

    return new_residual_history


def norm_position_residuals(cartesian_residual_history: dict) -> dict:

    epochs = list(cartesian_residual_history.keys())
    normed_residual_history = dict.fromkeys(epochs)
    number_of_iterations = int(len(cartesian_residual_history[epochs[0]]) / 3)

    for epoch in epochs:
        normed_residual_history[epoch] = np.zeros(number_of_iterations)
        for iteration in range(number_of_iterations):
            normed_residual_history[epoch][iteration] = norm(cartesian_residual_history[epoch][3*iteration:3*(iteration+1)])

    return normed_residual_history


def extract_orientation_residuals(residual_history: np.ndarray, observation_times: np.ndarray, number_of_iterations: float) -> dict:

    warnings.warn('(extract_orientation_residuals): Function not yet implemented. Returning None.')

    return


def get_orientation_statistical_indicators_evolution(residual_history: dict) -> dict:

    warnings.warn('(get_orientation_statistical_indicators_evolution): Function not yet implemented. Returning None.')

    return


def covariance_to_correlation(covariance_matrix: np.ndarray) -> np.ndarray:

    correlation_matrix = np.zeros_like(covariance_matrix)

    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i,j] = covariance_matrix[i,j] / np.sqrt(covariance_matrix[i,i]*covariance_matrix[j,j])

    return correlation_matrix
