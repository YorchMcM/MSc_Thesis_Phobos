import numpy as np
PI = np.pi
TWOPI = 2*PI


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


def array2result(array: np.ndarray) -> dict[float, np.ndarray]:

    keys = list(array[:,0])
    result = dict.fromkeys(keys)
    for idx in range(len(keys)):
        result[keys[idx]] = array[idx, 1:]

    return result


def make_between_zero_and_twopi(original: np.ndarray) -> np.ndarray:

    dim_num = len(original.shape)

    if dim_num == 1: return make_between_zero_and_twopi_single_dim(original)
    elif dim_num == 2: return make_between_zero_and_twopi_double_dim(original)
    else: raise ValueError('(make_between_zero_and_twopi): Invalid input array.')


def make_between_zero_and_twopi_single_dim(original: np.ndarray) -> np.ndarray:

    new = np.zeros_like(original)
    for idx in range(len(new)):
        new[idx] = original[idx]
        while new[idx] < 0.0: new[idx] = new[idx] + TWOPI
        while new[idx] >= TWOPI : new[idx] = new[idx] - TWOPI

    return new


def make_between_zero_and_twopi_double_dim(original: np.ndarray) -> np.ndarray:

    lengths = original.shape
    new = np.zeros_like(original)
    for idx0 in range(lengths[0]):
        for idx1 in range(lengths[1]):
            new[idx0, idx1] = original[idx0, idx1]
            while new[idx0, idx1] < 0.0: new[idx0, idx1] = new[idx0, idx1] + TWOPI
            while new[idx0, idx1] >= TWOPI: new[idx0, idx1] = new[idx0, idx1] - TWOPI

    return new


def extract_element_from_history(history: dict, index) -> dict:

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
