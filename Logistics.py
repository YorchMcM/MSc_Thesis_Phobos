import warnings

import numpy as np
from numpy import pi as PI
from tudatpy.io import save2txt
TWOPI = 2*PI


def norm_rows(array: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(array[k,:]) for k in range(array.shape[0])])


def norm_columns(array: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(array[:,k]) for k in range(array.shape[1])])


def norm_history(history: dict) -> dict:

    epoch_list = list(history.keys())
    normed_history = dict.fromkeys(epoch_list)
    for epoch in epoch_list:
        normed_history[epoch] = np.linalg.norm(history[epoch])

    return normed_history


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def str2vec(string: str, separator: str) -> np.ndarray:
    return np.array([float(element) for element in string.split(separator)])


def vectorize_matrix(matrix: np.ndarray) -> np.ndarray:

    n = len(matrix[:,0])

    vectorized_matrix = matrix[0,:]
    for idx in range(1,n):
        vectorized_matrix = np.concatenate((vectorized_matrix, matrix[idx,:]), 0)

    return vectorized_matrix


def unvectorize_matrix(vectorized_matrix: np.ndarray, dimensions: list[int]) -> np.ndarray:

    rows, columns = dimensions
    if rows*columns != len(vectorized_matrix):
        raise ValueError('(unvectorize_matrix): Dimensions provided incompatible with vectorized matrices encountered')

    unvectorized_matrix = np.zeros(dimensions)
    for idx in range(rows): unvectorized_matrix[idx,:] = vectorized_matrix[columns*idx:columns*(idx+1)]

    return unvectorized_matrix


def read_vector_history_from_file(file_name: str) -> dict:

    with open(file_name, 'r') as file: lines = file.readlines()
    keys = [float(line.split('\t')[0]) for line in lines]
    solution = dict.fromkeys(keys)
    for idx in range(len(keys)): solution[keys[idx]] = str2vec(lines[idx], '\t')[1:]

    return solution


def save_matrix_history_to_file(result: dict[float, np.ndarray], filename: str) -> None:

    key_list = list(result.keys())
    new_dict = dict.fromkeys(key_list)
    for key in key_list: new_dict[key] = vectorize_matrix(result[key])

    save2txt(new_dict, filename)

    return


def read_matrix_history_from_file(filename: str, dimensions: list[int]) -> dict[float, np.ndarray]:

    vectorized_history = read_vector_history_from_file(filename)
    key_list = list(vectorized_history.keys())
    unvectorized_history = dict.fromkeys(key_list)
    for key in key_list: unvectorized_history[key] = unvectorize_matrix(vectorized_history[key], dimensions)

    return unvectorized_history


def save_matrix_to_file(matrix: np.ndarray, filename: str) -> None:

    matrix_str = ''
    rows, columns = matrix.shape

    for row in range(rows):
        matrix_str = matrix_str + str(matrix[row,0])
        for column in range(1,columns):
            matrix_str = matrix_str + ' ' + str(matrix[row,column])
        matrix_str = matrix_str + '\n'

    with open(filename, 'w') as file: file.write(matrix_str)

    return


def read_matrix_from_file(file_name: str, dimensions: list[int]) -> np.ndarray[float]:

    result = np.zeros(dimensions)
    with open(file_name, 'r') as file: rows = file.readlines()
    if len(rows) != dimensions[0]: raise ValueError('(read_matrix_from_file): Provided dimensions do not match with encountered matrix.')

    for idx1 in range(dimensions[0]):
        components = rows[idx1].split(' ')
        for idx2 in range(dimensions[1]):
            result[idx1, idx2] = float(components[idx2])

    return result


def array2result(array: np.ndarray) -> dict[float, np.ndarray]:

    keys = list(array[:,0])
    result = dict.fromkeys(keys)
    for idx in range(len(keys)):
        result[keys[idx]] = array[idx, 1:]

    return result


def result2array(result: dict) -> np.ndarray:

    n = len(list(result.keys()))
    array_history = np.concatenate((np.array(list(result.keys())).reshape([n, 1]), np.vstack(list(result.values()))), 1)

    return array_history


def bring_history_inside_bounds(original: dict, lower_bound: float,
                                upper_bound: float, include: str = 'lower') -> np.ndarray:

    original_array = result2array(original)
    original_array[:,1:] = bring_inside_bounds(original_array[:,1:], lower_bound, upper_bound, include)
    new = array2result(original_array)

    return new


def bring_inside_bounds(original: np.ndarray, lower_bound: float,
                        upper_bound: float, include: str = 'lower') -> np.ndarray:

    reconvert = False

    if include not in ['upper', 'lower']:
        raise ValueError('(bring_inside_bounds): Invalid value for argument "include". Only "upper" and "lower" are allowed. Provided: ' + include)

    scalar_types = [float, np.float32, np.float64, np.float128]
    if type(original) in scalar_types:
        original = np.array([original])
        reconvert = True

    dim_num = len(original.shape)

    if dim_num == 1: to_return = bring_inside_bounds_single_dim(original, lower_bound, upper_bound, include)
    elif dim_num == 2: to_return = bring_inside_bounds_double_dim(original, lower_bound, upper_bound, include)
    else: raise ValueError('(bring_inside_bounds): Invalid input array.')

    if reconvert: to_return = to_return[0]

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


def remove_jumps(original: np.ndarray, jump_height: float, margin: float = 0.03) -> np.ndarray:

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
        # print('d = ' + str(d))
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


def extract_elements_from_history(history: dict, index) -> dict:

    if type(index) is int: index = [index]
    elif type(index) is list: pass
    else: raise TypeError('(extract_element_from_history): Illegal index type.')


    n = len(index)
    new_history = dict.fromkeys(list(history.keys()))
    for key in list(new_history.keys()):
        new_history[key] = np.zeros(n)
        k = 0
        for current_index in index:
            new_history[key][k] = history[key][current_index]
            k = k + 1

    return new_history


def get_epoch_elements_from_epoch(epoch: float) -> np.ndarray:

    day = epoch // 86400.0
    hours = (epoch % 86400.0) // 3600.0
    minutes = ((epoch % 86400.0) % 3600.0) // 60.0
    seconds = ((epoch % 86400.0) % 3600.0) % 60.0

    return day, hours, minutes, seconds


def set_axis_color(axis, side, color):

    axis.tick_params(axis='y', colors=color)
    axis.spines[side].set_color(color)
    axis.yaxis.label.set_color(color)

    return


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


def retrieve_ephemeris_files(model: str) -> tuple:

    if model == 'S':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/translation-s.eph'
        rotational_ephemeris_file = None
    elif model == 'A1':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/translation-a.eph'
        rotational_ephemeris_file = None
    elif model == 'A2':
        translational_ephemeris_file = None
        rotational_ephemeris_file = '/home/yorch/thesis/ephemeris/rotation-a.eph'
    elif model == 'B':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/translation-b.eph'
        rotational_ephemeris_file = '/home/yorch/thesis/ephemeris/rotation-b.eph'
    elif model == 'C':
        translational_ephemeris_file = '/home/yorch/thesis/ephemeris/translation-c.eph'
        rotational_ephemeris_file = '/home/yorch/thesis/ephemeris/rotation-c.eph'
    else:
        raise ValueError('Invalid observation model selected.')

    return translational_ephemeris_file, rotational_ephemeris_file
