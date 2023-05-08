import numpy as np
from numpy import pi as PI
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


def array2result(array: np.ndarray) -> dict[float, np.ndarray]:

    keys = list(array[:,0])
    result = dict.fromkeys(keys)
    for idx in range(len(keys)):
        result[keys[idx]] = array[idx, 1:]

    return result


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

    interval_length = upper_bound - lower_bound
    new = np.zeros_like(original)
    for idx in range(len(new)):
        new[idx] = original[idx]
        if include == 'lower':
            while new[idx] < lower_bound : new[idx] = new[idx] + interval_length
            while new[idx] >= upper_bound : new[idx] = new[idx] - interval_length
        if include == 'upper':
            while new[idx] <= lower_bound : new[idx] = new[idx] + interval_length
            while new[idx] > upper_bound : new[idx] = new[idx] - interval_length

    return new


def bring_inside_bounds_double_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    interval_length = upper_bound - lower_bound
    lengths = original.shape
    new = np.zeros_like(original)
    for idx0 in range(lengths[0]):
        for idx1 in range(lengths[1]):
            new[idx0, idx1] = original[idx0, idx1]
            if include == 'lower':
                while new[idx0, idx1] < lower_bound : new[idx0, idx1] = new[idx0, idx1] + interval_length
                while new[idx0, idx1] >= upper_bound : new[idx0, idx1] = new[idx0, idx1] - interval_length
            if include == 'upper':
                while new[idx0, idx1] <= lower_bound : new[idx0, idx1] = new[idx0, idx1] + interval_length
                while new[idx0, idx1] > upper_bound : new[idx0, idx1] = new[idx0, idx1] - interval_length

    return new


def remove_jumps(original: np.ndarray, jump_height: float, margin: float = 0.03) -> np.ndarray:

    dim_num = len(original.shape)

    if dim_num == 1: return remove_jumps_single_dim(original, jump_height)
    elif dim_num == 2: return remove_jumps_double_dim(original, jump_height)
    else: raise ValueError('(make_monotonic): Invalid input array.')


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