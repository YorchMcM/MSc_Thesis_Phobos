import warnings

import numpy as np
from numpy import pi as PI
from tudatpy.io import save2txt
TWOPI = 2*PI


def array2dict(array: np.ndarray) -> dict[float, np.ndarray]:

    '''

    This function converts an array into a dictionary. Each row of the array is an element of the dictionary. The first
    column is used as keys for the dictionary elements. The rest of the elements in each row are stored as a vector in
    the dictionary entry.

    :param array: The array that is to be converted into a dictionary.
    :return: The dictionary.
    '''

    return dict(zip(array[:,0], array[:,1:]))

'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                   BRING INSIDE BOUNDS FAMILY                                       ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


def bring_history_inside_bounds(original: dict, lower_bound: float,
                                upper_bound: float, include: str = 'lower') -> np.ndarray:

    '''

    It takes a dictionary whose values are either arrays or scalars, and brings all the numbers on the values to within
    a specified semi-open interval. It is the dictionary version of bring_inside_bounds. The operation is NOT applied on the keys.

    :param original: The original dictionary.
    :param lower_bound: The lower bound of the interval.
    :param upper_bound: The upper bound of the interval.
    :param include: The bound to be included.
    :return: The modified dictionary.
    '''

    if include not in ['upper', 'lower']:
        raise ValueError('(bring_history_inside_bounds): Invalid value of "include" input. Allowed ones are "upper" and'
                         ' "lower". Provided is "' + include + '".')

    original_array = dict2array(original)
    original_array[:,1:] = bring_inside_bounds(original_array[:,1:], lower_bound, upper_bound, include)
    new = array2dict(original_array)

    return new


def bring_inside_bounds(original: np.ndarray, lower_bound: float,
                        upper_bound: float, include: str = 'lower') -> np.ndarray:

    if include not in ['upper', 'lower']:
        raise ValueError('(bring_inside_bounds): Invalid value for argument "include". Only "upper" and "lower" are allowed. Provided: ' + include)

    scalar_types = [int, float, np.float32, np.float64, np.float128]
    if type(original) in scalar_types:
        to_return = bring_inside_bounds_scalar(original, lower_bound, upper_bound, include)
    else:

        dim_num = len(original.shape)

        if dim_num == 1: to_return = bring_inside_bounds_single_dim(original, lower_bound, upper_bound, include)
        elif dim_num == 2: to_return = bring_inside_bounds_double_dim(original, lower_bound, upper_bound, include)
        else: raise ValueError('(bring_inside_bounds): Invalid input array.')

    return to_return


def bring_inside_bounds_single_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    new = np.zeros_like(original)
    for idx in range(len(new)):
        new[idx] = bring_inside_bounds_scalar(original[idx], lower_bound, upper_bound, include)

    return new


def bring_inside_bounds_double_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    lengths = original.shape
    new = np.zeros_like(original)
    for idx0 in range(lengths[0]):
        for idx1 in range(lengths[1]):
            new[idx0, idx1] = bring_inside_bounds_scalar(original[idx0, idx1], lower_bound, upper_bound, include)

    return new


def bring_inside_bounds_scalar(original: float, lower_bound: float,
                               upper_bound: float, include: str = 'lower') -> float:

    if original == upper_bound or original == lower_bound:
        if include == 'lower':
            return lower_bound
        else:
            return upper_bound

    if lower_bound < original < upper_bound:
        return original

    center = (upper_bound + lower_bound) / 2.0

    if original < lower_bound: reflect = True
    else: reflect = False

    if reflect: original = 2.0*center - original

    dividend = original - lower_bound
    divisor = upper_bound - lower_bound
    remainder = dividend % divisor
    new = lower_bound + remainder

    if reflect: new = 2.0*center - new

    if new == lower_bound and include == 'upper': new = upper_bound
    if new == upper_bound and include == 'lower': new = lower_bound

    return new

'''
########################################################################################################################
########################################################################################################################
'''


def dict2array(result: dict) -> np.ndarray:

    n = len(list(result.keys()))
    array_history = np.concatenate((np.array(list(result.keys())).reshape([n, 1]), np.vstack(list(result.values()))), 1)

    return array_history


def extract_elements_from_history(history: dict, index: int | list[int]) -> dict:
    return dict(zip(np.array(list(history.keys())), np.vstack(list(history.values()))[:,index]))


def find_max_in_range(f: np.ndarray, range_of_interest: list[int]) -> float:

    temp = f[f[:,0] >= range_of_interest[0],:]
    temp = temp[temp[:,0] <= range_of_interest[1],:]

    return np.max(temp[:,1])


def matrix_result_to_column_array(matrix_dict: dict, column_to_extract: int) -> np.ndarray:

    epoch_list = list(matrix_dict.keys())
    rows = len(epoch_list)
    columns = len(matrix_dict[epoch_list[0]][:,column_to_extract]) + 1
    array_to_return = np.zeros([rows, columns])

    array_to_return[:,0] = np.array(epoch_list)
    for epoch_idx, epoch in enumerate(epoch_list): array_to_return[epoch_idx,1:] = matrix_dict[epoch][:,column_to_extract]

    return array_to_return


def matrix_result_to_entry_array(matrix_dict: dict, entry_to_extract: list[int]) -> np.ndarray:

    i, j = entry_to_extract
    epoch_list = list(matrix_dict.keys())
    rows = len(epoch_list)
    array_to_return = np.zeros([rows, 2])
    array_to_return[:, 0] = np.array(epoch_list)
    for epoch_idx, epoch in enumerate(epoch_list): array_to_return[epoch_idx,1] = matrix_dict[epoch][i,j]

    return array_to_return


def matrix_result_to_row_array(matrix_dict: dict, row_to_extract: int) -> np.ndarray:

    epoch_list = list(matrix_dict.keys())
    rows = len(epoch_list)
    columns = len(matrix_dict[epoch_list[0]][row_to_extract,:]) + 1
    array_to_return = np.zeros([rows, columns])

    array_to_return[:, 0] = np.array(epoch_list)
    for epoch_idx, epoch in enumerate(epoch_list): array_to_return[epoch_idx,1:] = matrix_dict[epoch][row_to_extract, :]

    return array_to_return


def read_matrix_history_from_file(filename: str, dimensions: list[int]) -> dict[float, np.ndarray]:

    vectorized_history = read_vector_history_from_file(filename)
    key_list = list(vectorized_history.keys())
    unvectorized_history = dict.fromkeys(key_list)
    for key in key_list: unvectorized_history[key] = unvectorize_matrix(vectorized_history[key], dimensions)

    return unvectorized_history


def read_matrix_from_file(file_name: str, dimensions: list[int]) -> np.ndarray[float]:

    result = np.zeros(dimensions)
    with open(file_name, 'r') as file: rows = file.readlines()
    if len(rows) != dimensions[0]: raise ValueError('(read_matrix_from_file): Provided dimensions do not match with encountered matrix.')

    for idx1 in range(dimensions[0]):
        components = rows[idx1].split(' ')
        for idx2 in range(dimensions[1]):
            result[idx1, idx2] = float(components[idx2])

    return result


def read_vector_history_from_file(file_name: str) -> dict:

    with open(file_name, 'r') as file: lines = file.readlines()
    content = np.vstack([str2vec(line, '\t') for line in lines])

    return dict(zip(content[:,0], content[:,1:]))


def reduce_columns(original: np.ndarray, n: int, loc: int = 0) -> np.ndarray:

    '''

    This function will reduce the number of columns of an array, keeping only one out of each n columns. If the number
    of columns of the original array is not divisible by n, it will eliminate trailing columns until it is.


    :param original:
    :param n:
    :param loc:
    :return:
    '''

    if n == 1:
        return original

    else:

        if loc >= n:
            raise ValueError('(reduce_columns): Incompatible arguments. Argument "loc" must be strictly smaller than '
                             'argument "n". Provided arguments are: n = ' + str(n) + ' and loc = ' + str(loc) + '.')

        else:
            original = original[:,:-(len(original.T) % n)]
            reduced = np.zeros([len(original), len(original.T) / n])
            for idx in range(len(reduced.T)):
                reduced[:,idx] = original[:,n*idx+loc]

            return reduced


def reduce_rows(original: np.ndarray, n: int, loc: int = 0) -> np.ndarray:
    '''

    This function will reduce the number of rows of an array, keeping only one out of each n columns. If the number
    of rows of the original array is not divisible by n, it will eliminate trailing rows until it is.


    :param original:
    :param n:
    :param loc:
    :return:
    '''

    if n == 1:
        return original

    else:

        if loc >= n:
            raise ValueError('(reduce_rows): Incompatible arguments. Argument "loc" must be strictly smaller than '
                             'argument "n". Provided arguments are: n = ' + str(n) + ' and loc = ' + str(loc) + '.')

        else:

            if len(original.shape) < 2:
                original = np.atleast_2d(original).T
                flatten = True
            else:
                flatten = False

            original = original[:-(len(original) % n),:]
            reduced = np.zeros([int(len(original) / n), len(original.T), ])
            for idx in range(len(reduced)):
                reduced[idx,:] = original[n*idx+loc,:]

            if flatten:
                reduced = reduced.flatten()

        return reduced


'''
########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                     REMOVE JUMPS FAMILY                                            ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################
'''


def remove_jumps(original: np.ndarray, jump_height: float, margin: float = 0.03) -> np.ndarray:

    '''

    CAUTION! This function iterates over EVERY SINGLE ENTRY of an array. For large arrays, this might take a while

    :param original:
    :param jump_height:
    :param margin:
    :return:
    '''

    dim_num = len(original.shape)

    if dim_num == 1: return remove_jumps_single_dim(original, jump_height, margin)
    elif dim_num == 2: return remove_jumps_double_dim(original, jump_height, margin)
    else: raise ValueError('(remove_jumps): Invalid input array.')


def remove_jumps_single_dim(original: np.ndarray, jump_height: float, margin: float = 0.03) -> np.ndarray:

    new = original.copy()
    u = 1.0 - margin
    l = -1.0 + margin
    for idx in range(len(new)-1):
        d = (new[idx+1] - new[idx]) / jump_height
        if d <= l: new[idx+1:] = new[idx+1:] + jump_height
        if d >= u: new[idx+1:] = new[idx+1:] - jump_height

    return new


def remove_jumps_double_dim(original: np.array, jump_height: float, margin: float = 0.03) -> np.ndarray:

    new = original.copy()
    u = 1.0 - margin
    l = -1.0 + margin
    for col in range(new.shape[1]):
        for row in range(new.shape[0]-1):
            d = ( new[row+1,col] - new[row,col] ) / jump_height
            if d <= l: new[row+1:,col] = new[row+1:,col] + jump_height
            if d >= u: new[row+1:,col] = new[row+1:,col] - jump_height

    return new

'''
########################################################################################################################
########################################################################################################################
'''


def retrieve_ephemeris_files(model: str, eph_subdir: str = '') -> tuple:

    if model == 'S':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'translation-s.eph'
        rotational_ephemeris_file = ''
    elif model == 'A1':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'translation-a.eph'
        rotational_ephemeris_file = ''
    elif model == 'A2':
        translational_ephemeris_file = ''
        rotational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'rotation-a.eph'
    elif model == 'B':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'translation-b.eph'
        rotational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'rotation-b.eph'
    elif model == 'C':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'translation-c.eph'
        rotational_ephemeris_file = '/home/yorch/thesis/ephemeris/' + eph_subdir + 'rotation-c.eph'
    else:
        raise ValueError('Invalid observation model selected.')

    return translational_ephemeris_file, rotational_ephemeris_file


def search_for_first_and_last_zeros(f: np.ndarray) -> list[int]:

    to_return = [0, 0]
    N = len(f)

    if f.shape[1] == 1:
        f = f.flatten()
    else:
        f = f[:,0].flatten()

    idx = -1
    found = False
    while not found:
        idx += 1
        zero = f[idx]*f[idx+1] <= 0.0
        if zero:
            take_second = np.abs(f[idx+1]) < np.abs(f[idx])
            found = True
            to_return[0] = idx + take_second
        elif idx == N-2:
            warnings.warn('(search_for_first_and_last_zeros): First zero not found. Things might break later.')
            break

    idx = N
    found = False
    while not found:
        idx -= 1
        zero = f[idx]*f[idx-1] <= 0.0
        if zero:
            take_second = np.abs(f[idx-1]) < np.abs(f[idx])
            found = True
            to_return[1] = idx - take_second
        elif idx == 1:
            warnings.warn('(search_for_first_and_last_zeros): Last zero not found. Things might break later.')
            break

    return to_return

def seconds_since_j2000_to_days_hours_minutes_seconds(epoch: float) -> np.ndarray:

    day = epoch // 86400.0
    hours = (epoch % 86400.0) // 3600.0
    minutes = ((epoch % 86400.0) % 3600.0) // 60.0
    seconds = ((epoch % 86400.0) % 3600.0) % 60.0

    return day, hours, minutes, seconds


def str2vec(string: str, separator: str) -> np.ndarray:
    return np.array([float(element) for element in string.split(separator)])


def unvectorize_matrix(vectorized_matrix: np.ndarray, dimensions: list[int]) -> np.ndarray:

    rows, columns = dimensions
    if rows*columns != len(vectorized_matrix):
        raise ValueError('(unvectorize_matrix): Dimensions provided incompatible with vectorized matrices encountered')

    unvectorized_matrix = np.zeros(dimensions)
    for idx in range(rows): unvectorized_matrix[idx,:] = vectorized_matrix[columns*idx:columns*(idx+1)]

    return unvectorized_matrix


def vectorize_matrix(matrix: np.ndarray) -> np.ndarray:

    n = len(matrix[:,0])

    vectorized_matrix = matrix[0,:]
    for idx in range(1,n):
        vectorized_matrix = np.concatenate((vectorized_matrix, matrix[idx,:]), 0)

    return vectorized_matrix


def write_matrix_history_to_file(result: dict[float, np.ndarray], filename: str) -> None:

    key_list = list(result.keys())
    new_dict = dict.fromkeys(key_list)
    for key in key_list: new_dict[key] = vectorize_matrix(result[key])

    save2txt(new_dict, filename)

    return


def write_matrix_to_file(matrix: np.ndarray, filename: str) -> None:

    matrix_str = ''
    rows, columns = matrix.shape

    for row in range(rows):
        matrix_str = matrix_str + str(matrix[row,0])
        for column in range(1,columns):
            matrix_str = matrix_str + ' ' + str(matrix[row,column])
        matrix_str = matrix_str + '\n'

    with open(filename, 'w') as file: file.write(matrix_str)

    return
