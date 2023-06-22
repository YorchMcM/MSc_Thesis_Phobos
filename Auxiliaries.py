# NATIVE IMPORTS
import sys
import os
import warnings
from datetime import datetime

from time import time
from cycler import cycler

if '/home/yorch/tudat-bundle/cmake-build-release/tudatpy' not in sys.path:
    sys.path.insert(0, '/home/yorch/tudat-bundle/cmake-build-release/tudatpy')

from astromath import *
from Logistics import *

# TUDAT IMPORTS

# Domestic imports (i.e. only used in this file)
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
from tudatpy.kernel.astro.element_conversion import quaternion_entries_to_rotation_matrix as quat2mat
from tudatpy.kernel.astro.element_conversion import cartesian_to_keplerian, true_to_mean_anomaly, semi_major_axis_to_mean_motion

# Public imports (i.e. required by all other scripts importing this module but not necessarily used here)
from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.util import compare_results
from tudatpy.plotting import trajectory_3d
from tudatpy.io import save2txt

import matplotlib.font_manager as fman
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')
plt.rcParams['axes.prop_cycle'] = cycler('color', ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                                                   '#77AC30', '#4DBEEE', '#A2142F', '#7f7f7f', '#bcbd22', '#17becf'])
plt.rcParams['lines.markersize'] = 6.0

spice.load_standard_kernels()

'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                           MATH THAT REQUIRES TUDAT FUNCTIONS                                       ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def create_vector_interpolator(data: dict[float, np.ndarray]) -> interpolators.OneDimensionalInterpolatorVector:

    lagrange_settings = interpolators.lagrange_interpolation(number_of_points = 8)
    interpolator = interpolators.create_one_dimensional_vector_interpolator(data, lagrange_settings)

    return interpolator


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


def quaternion_to_matrix_history(quaternion_history: dict) -> dict:

    epochs_list = list(quaternion_history.keys())
    rotation_matrix_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        rotation_matrix_history[key] = quat2mat(quaternion_history[key])

    return rotation_matrix_history


def average_mean_motion_over_integer_number_of_orbits(keplerian_history: dict, gravitational_parameter: float) -> float:

    mean_motion_history = mean_motion_history_from_keplerian_history(keplerian_history, gravitational_parameter)
    periapses = get_periapses(keplerian_history)
    first_periapsis = periapses[0][0]
    last_periapsis = periapses[-1][0]
    mean_motion_over_integer_number_of_orbits = np.array(list(mean_motion_history.values())[first_periapsis:last_periapsis])

    return np.mean(mean_motion_over_integer_number_of_orbits), len(periapses)


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                          LOGISTICS THAT REQUIRE TUDAT FUNCTIONS                                    ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def get_ephemeris_from_file(filename: str) -> environment_setup.ephemeris.EphemerisSettings:

    trajectory = read_vector_history_from_file(filename)
    imposed_trajectory = extract_elements_from_history(trajectory, [0, 1, 2, 3, 4, 5])
    phobos_ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')

    return phobos_ephemerides


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


def reduce_gravity_field(settings: numerical_simulation.environment_setup.gravity_field.GravityFieldSettings) -> None:

    new_sine_coefficients = np.zeros_like(settings.normalized_sine_coefficients)
    new_cosine_coefficients = np.zeros_like(settings.normalized_cosine_coefficients)

    new_cosine_coefficients[0,0] = 1.0
    new_cosine_coefficients[2,0] = settings.normalized_cosine_coefficients[2,0]
    new_cosine_coefficients[2,2] = settings.normalized_cosine_coefficients[2,2]

    settings.normalized_sine_coefficients = new_sine_coefficients
    settings.normalized_cosine_coefficients = new_cosine_coefficients

    return


def save_initial_states(damping_results: numerical_simulation.propagation.DampedInitialRotationalStateResults,
                        filename: str) -> None:
    
    initial_states_str = '\n'
    for iteration in range(len(damping_results.forward_backward_states)):
        initial_forward_state = list(damping_results.forward_backward_states[iteration][0].values())[0]
        initial_backward_state = list(damping_results.forward_backward_states[iteration][-1].values())[0]
        initial_states_str = initial_states_str + ' ' + str(iteration) + ' F ' + str(initial_forward_state) + '\n'
        initial_states_str = initial_states_str + ' ' + str(iteration) + ' B ' + str(initial_backward_state) + '\n'
        
    with open(filename, 'w') as file: file.write(initial_states_str)
    
    return


def get_parameter_set(estimated_parameters: list[str],
                      bodies: numerical_simulation.environment.SystemOfBodies,
                      propagator_settings: propagation_setup.propagator.PropagatorSettings | None = None,
                      return_only_settings_list: bool = True) -> tuple:

    parameter_settings = []
    parameters_str = ''

    if propagator_settings is not None:
        parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
        parameters_str = parameters_str + '\t- Initial state\n'

    if 'A' in estimated_parameters:
        warnings.warn('Libration amplitude selected as parameter to estimate, but not exposed to Python yet.')
        # AQUÍ ME FALTA EXPONER LA LIBRATION AMPLITUDE COMO ESTIMATABLE PARAMETER (CREO)
        pass
    if 'C20' in estimated_parameters and 'C22' in estimated_parameters:
        parameter_settings = parameter_settings + [
            estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 0), (2, 2)])]
        parameters_str = parameters_str + '\t- C20\n'
        parameters_str = parameters_str + '\t- C22\n'
    elif 'C20' in estimated_parameters:
        parameter_settings = parameter_settings + [
            estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 0)])]
        parameters_str = parameters_str + '\t- C20\n'
    elif 'C22' in estimated_parameters:
        parameter_settings = parameter_settings + [
            estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 2)])]
        parameters_str = parameters_str + '\t- C22\n'

    if return_only_settings_list:
        return parameter_settings, parameters_str
    else:
        parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
        return parameters_to_estimate, parameters_str


class EstimationSettings:

    def __init__(self, settings_file: str):

        self.source_file = settings_file
        self.estimation_settings = dict()
        self.observation_settings = dict()
        self.ls_convergence_settings = dict()
        self.test_functionalities = dict()
        self.execution_settings = dict()

        self.read_settings_from_file()
        self.run_diagnostics_on_settings()

        return

    def read_settings_from_file(self) -> None:

        with open(self.source_file, 'r') as file: all_lines = [line for line in file.readlines() if line != '\n']

        dicts = [self.estimation_settings, self.observation_settings, self.ls_convergence_settings, self.test_functionalities, self.execution_settings]
        dict_idx = -1
        for line in all_lines:
            if '#####' in line:
                continue
            if '####' in line:
                dict_idx = dict_idx + 1
                continue
            if ':' in line:

                dict_key = line.split(':')[0].removeprefix('· ').lower()
                dict_value = line.split('#')[0]
                while dict_value[-1] == ' ' or dict_value[-1] == '\t': dict_value = dict_value[:-1]  # Remove all trailing spaces or tabs
                dict_value = dict_value.split(':')[1].removesuffix('\n').replace('\t', '')
                while dict_value[0] == ' ' or dict_value[0] == '\t': dict_value = dict_value[1:]  # Remove all preceding spaces or tabs

                # A SPECIAL CASE HERE IS THAT THERE IS A UNIT IN THE INPUT (MINUTE, DAY, YEAR, ...)
                if 'YEAR' in dict_value or 'DAY' in dict_value or 'HOUR' in dict_value or 'MINUTE' in dict_value or \
                    'SECOND' in dict_value:

                    value = float(dict_value.removeprefix(' ').split(' ')[0])
                    unit = dict_value.split(' ')[1]

                    if 'SECOND' in unit: factor = 1.0
                    elif 'MINUTE' in unit: factor = 60.0
                    elif 'HOUR' in unit: factor = 3600.0
                    elif 'DAY' in unit: factor = 86400.0
                    elif 'YEAR' in unit: factor = 86400*365.25

                    value = value * factor

                    if 'OFFSET' in dict_value:
                        if dict_key == 'epoch of first observation':
                            self.observation_settings[dict_key] = self.estimation_settings['initial estimation epoch'] + value
                        if dict_key == 'epoch of last observation':
                            self.observation_settings[dict_key] = self.estimation_settings['initial estimation epoch'] + \
                                                                  self.estimation_settings['duration of estimated arc'] + value
                    else:
                        dicts[dict_idx][dict_key] = value

                    # The work with this particular line is done. We can continue directly to the next.
                    continue

                # THE REST OF THE CASES IS FLOATS, STRINGS OR SEQUENCES THEREOF SEPARATED BY ", "
                try:
                    dict_value = float(dict_value)  # We try and see if it can be converted into a float.
                except:
                    if ',' not in dict_value:  # In this case, the value is not a float but also not a sequence, therefore just a string.
                        if dict_value == 'True' or dict_value == 'False':  # It it represents a boolean, it should be converted.
                            dict_value = eval(dict_value)
                        else:  # Otherwise, it should be left as is.
                            pass
                    else:
                        dict_value = dict_value.split(', ')
                        try:
                            dict_value = np.array([float(element) for element in dict_value])  # This is the case in which they are all numbers, and they represent an array.
                        except:
                            pass  # In this case, the value should be just the list of strings we got above with the split command

                # At this point, we have the correct dict value. We just assign it to the entry.
                dicts[dict_idx][dict_key] = dict_value

        self.ls_convergence_settings['maximum number of iterations'] = \
            int(self.ls_convergence_settings['maximum number of iterations'])
        self.ls_convergence_settings['number of iterations without improvement'] = \
            int(self.ls_convergence_settings['number of iterations without improvement'])

        return

    def run_diagnostics_on_settings(self) -> None:

        if self.estimation_settings['estimation model'] not in ['S', 'A1', 'A2', 'B', 'C']:
            raise ValueError('Invalid estimation model.')

        if type(self.observation_settings['observation type']) is list:
            raise ValueError('You chose more than one type of observation. Don\'t do that just yet.')

        if 'initial state' not in self.estimation_settings['estimated parameters']:
            warnings.warn('The initial state has not been selected as an estimated parameter. That is a bad idea and it'
                          ' will be estimated anyway.')

        model = self.estimation_settings['estimation model']
        obs = self.observation_settings['observation type']
        libration_estimated = 'A' in self.estimation_settings['estimated parameters']

        if (model == 'A1' and obs == 'orientation') or (model == 'A2' and obs == 'position'):
            raise ValueError('The selected estimation model (' + model + ') is incompatible with the selected '
                                                                         'observation type (' + obs + ').')

        if model in ['A2', 'B', 'C'] and libration_estimated:
            raise ValueError('The selected estimation model (' + model + ') is incompatible with the selected '
                                                                         'estimation of the libration amplitude')

        t0_e = self.estimation_settings['initial estimation epoch']
        tf_e = self.estimation_settings['initial estimation epoch'] + self.estimation_settings['duration of estimated arc']
        t0_o = self.observation_settings['epoch of first observation']
        tf_o = self.observation_settings['epoch of last observation']
        params = self.estimation_settings['estimated parameters']

        if tf_o <= t0_o:
            raise ValueError('The epoch of the last observation is lower than (or equal to) the epoch of the first '
                             'observation.')

        if tf_e <= tf_o:
            warnings.warn('Your initial estimation epoch is greater than (or equal to) your final estimation epoch. '
                          'This will cause the propagation of your equations of motion and variational equations to run'
                          ' backwards. The "initial state" will be that at the final epoch.')

        if t0_o < t0_e or tf_o > tf_e:
            warnings.warn('Some of your observations are outside your estimation period. They will be ignored.')

        if 'A' in params and 'C20' in params: ill_posedness = True
        elif 'A' in params and 'C22': ill_posedness = True
        elif 'C20' in params and 'C22': ill_posedness = True
        elif 'A' in params and 'C20' in params and 'C22' in params: ill_posedness = True
        else: ill_posedness = False
        if ill_posedness:
            warnings.warn('Your estimated parameters include combinations that are known to be prone to ill-posedness of'
                          ' the least squares problem. Expect large condition numbers.')

        if not self.test_functionalities['test mode'] and self.estimation_settings['estimation model'] == self.observation_settings['observation model']:
            warnings.warn('Test mode was not selected, yet the observation and estimation models are identical. First '
                          'estimation iteration already departs from truth.')

        return

    def get_estimation_type(self) -> str | None:

        estimation_type = None
        obs = self.observation_settings['observation type']
        params = self.estimation_settings['estimated parameters']
        model = self.estimation_settings['estimation model']
        if model == 'A1':
            if params == ['initial state', 'C20', 'C22']:
                estimation_type = 'alpha'
            elif params == ['initial state', 'A']:
                estimation_type = 'bravo'
            elif params == ['initial state', 'A', 'C20', 'C22']:
                estimation_type = 'charlie'
            else: pass

        if model == 'B' and obs == 'position':
            estimation_type = 'delta'

        if model == 'A2':
            estimation_type = 'echo'

        if model == 'B' and obs == 'orientation':
            estimation_type = 'foxtrot'

        return estimation_type

    def get_initial_state(self, type: str) -> np.ndarray:

        type = type.lower()
        if type not in ['true', 'estimation']:
            raise ValueError('Invalid type of initial state. Only accepted options are "true" and "estimation".')

        if type == 'estimation':
            model = self.estimation_settings['estimation model']
        else:
            model = self.observation_settings['observation model']

        if model != 'A2':
            initial_translational_state = self.get_initial_translational_state(type, model)
        if model in ['A2', 'B', 'C']:
            initial_rotational_state = self.get_initial_rotational_state(type, model)

        if model in ['S', 'A1']:
            initial_state = initial_translational_state
        elif model == 'A2':
            initial_state = initial_rotational_state
        else:
            initial_state = np.concatenate((initial_translational_state, initial_rotational_state))

        return initial_state

    def get_initial_translational_state(self, type: str, model: str | None = None) -> np.ndarray:

        if self.estimation_settings['estimation model'] == 'A2':
            raise ValueError('You\'re in the wrong place my dude')

        if model is None:
            if type == 'estimation':
                model = self.estimation_settings['estimation model']
            else:
                model = self.observation_settings['observation model']

        ephemeris, trash = retrieve_ephemeris_files(model)
        initial_state = create_vector_interpolator(read_vector_history_from_file(ephemeris)).interpolate(
            self.estimation_settings['initial estimation epoch']
        )

        if type == 'estimation' and self.test_functionalities['test mode']:
            pos_pert = self.test_functionalities['initial position perturbation']
            vel_pert = self.test_functionalities['initial velocity perturbation']
            pert = np.concatenate((pos_pert, vel_pert))
            if self.test_functionalities['apply perturbation in rsw']:
                RSW_R_I = inertial_to_rsw_rotation_matrix(initial_state)
                R = np.concatenate((np.concatenate((RSW_R_I, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), RSW_R_I), 1)), 0)
                pert = R.T @ pert
            initial_state = initial_state + pert

        return initial_state

    def get_initial_rotational_state(self, type: str, model: str | None = None) -> np.ndarray:

        if self.estimation_settings['estimation model'] in ['S', 'A1']:
            raise ValueError('You\'re in the wrong place my dude')

        if model is None:
            if type == 'estimation':
                model = self.estimation_settings['estimation model']
            else:
                model = self.observation_settings['observation model']

        trash, ephemeris = retrieve_ephemeris_files(model)
        initial_state = create_vector_interpolator(read_vector_history_from_file(ephemeris)).interpolate(
            self.estimation_settings['initial estimation epoch']
        )

        if type == 'estimation' and self.test_functionalities['test mode']:
            ori_pert = self.test_functionalities['initial orientation perturbation']
            ome_pert = self.test_functionalities['initial angular velocity perturbation']
            pert = np.concatenate((ori_pert, ome_pert))
            initial_state = initial_state + pert

        return initial_state


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                     FULL-ON TUDAT SHIT                                             ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def get_gravitational_field(frame_name: str,
                            field_type: str = 'QUAD') -> environment_setup.gravity_field.GravityFieldSettings:

    # The gravitational field implemented here is that in Le Maistre et al. (2019).

    datadir = os.getcwd() + '/Normalized gravity fields/'
    cosines_file = datadir + 'cosines Le Maistre.txt'
    sines_file = datadir + 'sines Le Maistre.txt'

    phobos_gravitational_parameter = 1.06e16*constants.GRAVITATIONAL_CONSTANT
    phobos_reference_radius = 14e3

    phobos_normalized_cosine_coefficients = read_matrix_from_file(cosines_file, [5,5])
    phobos_normalized_sine_coefficients = read_matrix_from_file(sines_file, [5, 5])

    settings_to_return = environment_setup.gravity_field.spherical_harmonic(
        phobos_gravitational_parameter,
        phobos_reference_radius,
        phobos_normalized_cosine_coefficients,
        phobos_normalized_sine_coefficients,
        associated_reference_frame = frame_name)

    if field_type == 'QUAD': reduce_gravity_field(settings_to_return)

    return settings_to_return


def perform_propagator_checks(bodies: numerical_simulation.environment.SystemOfBodies,
                              model: str,
                              initial_epoch: float,
                              initial_state: np.ndarray,
                              simulation_time: float,
                              dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [])\
    -> None:

    # Check if the model is valid or not.
    if model not in ['S', 'A1', 'A2', 'B', 'C']:
        raise ValueError('The model you provided is not supported.')

    # Make sure that the libration corresponds to the model (i.e. 0 for model S and 1.1 for model A1).
    if bodies.get('Phobos').rotation_model.libration_calculator is not None: libration = True
    else: libration = False
    if model in ['A2', 'B', 'C']: correct_rotation = True
    elif model == 'S' and not libration: correct_rotation = True
    elif model == 'A1' and libration: correct_rotation = True
    else: correct_rotation = False
    if not correct_rotation:
        warnings.warn('The rotation of Phobos does not correspond to the selected model')

    # Make sure that the gravity field corresponds to the selected model
    quad_gravity = sum(sum(bodies.get('Phobos').gravity_field_model.sine_coefficients)) == 0.0
    if model in ['S', 'A1', 'A2', 'B'] and quad_gravity: correct_gravity = True
    elif model == 'C' and not quad_gravity: correct_gravity = True
    else: correct_gravity = False
    if not correct_gravity:
        warnings.warn('The gravity field of Phobos does not correspond to the selected model.')

    # Check, for one, that the state vector is compatible with the model, and for two, that ephemeris are correct.
    if model in ['S', 'A1'] and len(initial_state) == 6: correct_initial_state = True
    elif model == 'A2' and len(initial_state) == 7: correct_initial_state = True
    elif model in ['B', 'C'] and len(initial_state) == 13: correct_initial_state = True
    else: correct_initial_state = False
    if not correct_initial_state:
        raise ValueError('The state vector provided is not compatible with the model selected.')

    # If the selected model is A2, the ephemeris of Phobos will be used. Make sure there is no dangerous (inter/extra)polation
    if model == 'A2'and type(bodies.get('Phobos').ephemeris) is numerical_simulation.environment.TabulatedEphemeris:
        out_of_bounds = initial_epoch == bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[0] > \
                        initial_epoch or bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[-1] < \
                        initial_epoch + simulation_time
        if out_of_bounds:
            warnings.warn(
                'Your choice of model will make use of the ephemeris of Phobos. However, the available ephemeris does not'
                ' cover the entire domain of the selected simulation time. Values for the state will be extrapolated '
                'for those times not covered by your ephemeris. These results are not to be trusted.' )
        interpolation = initial_epoch == bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[3] > \
                        initial_epoch or bodies.get('Phobos').ephemeris.interpolator.get_independent_values()[-4] < \
                        initial_epoch + simulation_time
        if interpolation:
            warnings.warn(
                'Your choice of model will make use of the ephemeris of Phobos. Your simulation starts exactly'
                ' on the first data point of the available ephemeris. It is likely that your integrator will '
                'have to interpolate between the first 4 data points provided in the ephemeris. Interpolation '
                'in this range is deficient. Results are not to be trusted.' )

    return


def get_translational_dynamics_propagator(bodies: numerical_simulation.environment.SystemOfBodies,
                                          bodies_to_propagate: list[str],
                                          central_bodies: list[str],
                                          initial_epoch: float,
                                          initial_state: np.ndarray,
                                          integrator_settings: propagation_setup.integrator.IntegratorSettings,
                                          termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
                                          dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = []) \
        -> propagation_setup.propagator.TranslationalStatePropagatorSettings:

    if sum(sum(abs(bodies.get('Phobos').gravity_field_model.sine_coefficients))) > 0.0: shdo = 4  # shdo = spherical harmonic degree and order
    else: shdo = 2

    # ACCELERATION SETTINGS
    third_body_force = propagation_setup.acceleration.point_mass_gravity()
    acceleration_settings_on_phobos = dict(Mars=[propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, shdo, shdo)],
                                           Sun=[third_body_force],
                                           Earth=[third_body_force],
                                           Deimos=[third_body_force],
                                           Jupiter=[third_body_force])
    acceleration_settings = {'Phobos': acceleration_settings_on_phobos}
    acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    # PROPAGATION SETTINGS
    propagator_settings = propagation_setup.propagator.translational(central_bodies,
                                                                     acceleration_model,
                                                                     bodies_to_propagate,
                                                                     initial_state,
                                                                     initial_epoch,
                                                                     integrator_settings,
                                                                     termination_condition,
                                                                     output_variables = dependent_variables)

    return propagator_settings


def get_rotational_dynamics_propagator(bodies: numerical_simulation.environment.SystemOfBodies,
                                       bodies_to_propagate: list[str],
                                       central_bodies: list[str],
                                       initial_epoch: float,
                                       initial_state: np.ndarray,
                                       integrator_settings: propagation_setup.integrator.IntegratorSettings,
                                       termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
                                       dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = []) \
        -> propagation_setup.propagator.RotationalStatePropagatorSettings:

    if sum(sum(abs(bodies.get('Phobos').gravity_field_model.sine_coefficients))) > 0.0: shdo = 4  # shdo = spherical harmonic degree and order
    else: shdo = 2

    # TORQUE SETTINGS
    torque_on_phobos = propagation_setup.torque.spherical_harmonic_gravitational(shdo,shdo)
    torque_settings_on_phobos = dict(Mars=[torque_on_phobos],
                                     Sun=[torque_on_phobos],
                                     Earth=[torque_on_phobos],
                                     Deimos=[torque_on_phobos],
                                     Jupiter=[torque_on_phobos])
    torque_settings = {'Phobos': torque_settings_on_phobos}
    torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

    # PROPAGATION SETTINGS
    propagator_settings = propagation_setup.propagator.rotational(torque_model,
                                                                  bodies_to_propagate,
                                                                  initial_state,
                                                                  initial_epoch,
                                                                  integrator_settings,
                                                                  termination_condition,
                                                                  output_variables = dependent_variables)

    return propagator_settings


def get_undamped_initial_state_at_epoch(bodies: numerical_simulation.environment.SystemOfBodies,
                                        model: str,
                                        epoch: float,
                                        phobos_mean_rotational_rate: float = 0.000228035245) -> np.ndarray:

    if model not in ['A2', 'B', 'C']:
        raise ValueError('Model does not require a call to this function.')

    phobos_ephemeris = bodies.get('Phobos').ephemeris
    if type(phobos_ephemeris) is spice.SpiceEphemeris:
        translational_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'None', epoch)
    elif type(phobos_ephemeris) is numerical_simulation.environment.TabulatedEphemeris:
        translational_state = phobos_ephemeris.interpolator.interpolate(epoch)
    else:
        raise TypeError('Unsupported ephemeris.')

    phobos_rotation = bodies.get('Phobos').rotation_model
    omega = np.array([0, 0, phobos_mean_rotational_rate])
    if type(phobos_rotation) is numerical_simulation.environment.SynchronousRotationalEphemeris:
        synchronous_orientation = inertial_to_rsw_rotation_matrix(translational_state).T
        synchronous_orientation[:,:2] = -1.0*synchronous_orientation[:,:2]
        synchronous_orientation = mat2quat(synchronous_orientation)
        rotational_state = np.concatenate((synchronous_orientation, omega), 0)
    elif type(phobos_rotation) is numerical_simulation.environment.TabulatedRotationalEphemeris:
        rotational_state = (phobos_rotation.interpolator.interpolate(epoch))
    else:
        raise TypeError('Unsupported rotation.')

    if model == 'A2': full_state = rotational_state
    else: full_state = np.concatenate((translational_state, rotational_state), 0)

    return full_state


def get_list_of_dependent_variables(model: str, bodies: numerical_simulation.environment.SystemOfBodies) \
        -> list[propagation_setup.dependent_variable.PropagationDependentVariables]:

    if model in ['S', 'A1']:
        mutual_spherical = propagation_setup.acceleration.mutual_spherical_harmonic_gravity_type
        mars_acceleration_dependent_variable = propagation_setup.dependent_variable.single_acceleration_norm(mutual_spherical, 'Phobos', 'Mars')
        dependent_variables_to_save = [ propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos'),  # 0, 1, 2
                                        propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                        propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),  # 6, 7, 8, 9, 10, 11
                                        propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                        acceleration_norm_from_body_on_phobos('Sun'), # 15
                                        acceleration_norm_from_body_on_phobos('Earth'),  # 16
                                        mars_acceleration_dependent_variable,  # 17
                                        acceleration_norm_from_body_on_phobos('Deimos'),  # 18
                                        acceleration_norm_from_body_on_phobos('Jupiter')  # 19
                                        ]

    elif model in ['A2', 'B', 'C']:
        euler_angles_wrt_mars_equator_dependent_variable = \
            propagation_setup.dependent_variable.custom_dependent_variable(MarsEquatorOfDate(bodies).get_euler_angles_wrt_mars_equator, 3)
        dependent_variables_to_save = [euler_angles_wrt_mars_equator_dependent_variable,  # 0, 1, 2
                                       propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                       propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),  # 6, 7, 8, 9, 10, 11
                                       propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                       propagation_setup.dependent_variable.relative_position('Phobos', 'Mars')  # 15, 16, 17
                                       ]

    else: dependent_variables_to_save = []

    return dependent_variables_to_save


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


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                            ENVIRONMENT AND PROPAGATION SETUPS                                      ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def get_solar_system(model: str,
                     translational_ephemeris_file: str | None = None,
                     rotational_ephemeris_file: str | None = None,
                     **additional_inputs) -> numerical_simulation.environment.SystemOfBodies:

    '''
    This function will create the "bodies" object, just so we don't have all this lines in all models. Phobos is to be
    created from scratch. All its properties are created according to the model provided to the function through the
    "model" argument: fully synchronous (S), model A1 (A1), model A2 (A2), model B (B) or model C (C). More information
    on the environment associated to each model can be found in each model's file.

    For the integration of the rotational dynamics, Phobos needs to have an associated ephemeris. Similarly, when
    creating the bodies for use inside an estimation scheme, the observations will be simulated according to the bodies'
    environment (ephemeris and rotation model). In these scenarios, a file containing translational/rotational state
    history can be provided through the translational_ephemeris_file/rotational_ephemeris_file arguments.

    Additionally, as an extra, the **additional inputs can gather other personalized information. For now, the only
    supported option is that of selecting the libration amplitude. The named argument should be called
    "libration_amplitude" and should be passed in DEGREES.

    :param model: The model to be used. It can be S, A1, A2, B or C.
    :param translational_ephemeris_file: The name of a file containing a history of state vectors to assign to Phobos.
    :param rotational_ephemeris_file: The name of a file containing a history of state vectors to assign to Phobos.
    :param **additional_inputs: Other stuff. For now, the only thign supported is "libration_amplitude".
    :return: bodies
    '''

    if model not in ['S', 'A1', 'A2', 'B', 'C']:
        raise ValueError('Model provided is invalid.')

    if model == 'A2' and translational_ephemeris_file is None:
        warnings.warn('The model you selected requires a translational ephemeris. No ephemeris file was provided. Reverting to default spice ephemeris.')

    # WE FIRST CREATE MARS.
    bodies_to_create = ["Sun", "Earth", "Mars", "Deimos", "Jupiter"]
    global_frame_origin = "Mars"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

    # WE THEN CREATE PHOBOS USING THE INPUTS.
    body_settings.add_empty_settings('Phobos')
    # Ephemeris model. If no file is provided, defaults to spice ephemeris.
    if translational_ephemeris_file is None:
        body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
    else:
        imposed_trajectory = read_vector_history_from_file(translational_ephemeris_file)
        body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
    # Rotational model. If no file is provided, defaults to synchronous rotation.
    if rotational_ephemeris_file is None:
        body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
    else:
        imposed_rotation = read_vector_history_from_file(rotational_ephemeris_file)
        body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.tabulated(imposed_rotation,
                                                                                                         'J2000',
                                                                                                         'Phobos_body_fixed')
    # Gravity field.
    if model in ['S', 'A1', 'A2', 'B']: field_type = 'QUAD'
    else: field_type = 'FULL'
    body_settings.get('Phobos').gravity_field_settings = get_gravitational_field('Phobos_body_fixed', field_type)
    I = 0.43  # Mean moment of inertia taken from Rambaux 2012 (no other number found anywhere else)
    body_settings.get('Phobos').gravity_field_settings.scaled_mean_moment_of_inertia = I

    # AND LASTLY THE LIST OF BODIES IS CREATED.
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # The libration can only be assigned to the rotation model that is created from the settings when the SystemOfBodies object is created.
    if model == 'A1':
        if 'libration_amplitude' in additional_inputs: libration_amplitude = additional_inputs['libration_amplitude']
        else: libration_amplitude = 1.1
        # scaled libration amplitude = libration amplitude / eccentricity
        scaled_amplitude = np.radians(libration_amplitude) / 0.015034167790105173
        bodies.get('Phobos').rotation_model.libration_calculator = numerical_simulation.environment.DirectLongitudeLibrationCalculator(scaled_amplitude)

    return bodies


def get_propagator_settings(model: str,
                            bodies: numerical_simulation.environment.SystemOfBodies,
                            initial_epoch: float,
                            initial_state: np.ndarray,
                            simulation_time: float,
                            dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [])\
    -> propagation_setup.propagator.PropagatorSettings:

    '''
    This function will create the propagator settings for all models. The type of dynamics and therefore the
    characteristics of the resulting propagator settings object are fully defined by the model passed through the
    "model" argument: synchronous (S), model A1 (A1), model A2 (A2), model B (B) or model C (C). More information
    on the dynamcis associated to each model can be found in each model's file.

    The other arguments are mainly controls for the simulation: initial epoch, initial state and simulation duration.
    These are user inputs that will be assumed to be interest to change, and are therefore left for the user to define
    them in a higher level module.

    The termination condition for the simulation will always be a time termination one. That is hard coded inside the
    present function.

    :param bodies: The system of bodies.
    :param model: The model to get the propagator settings for. It can be 'S', 'A1', 'A2', 'B' or 'C'.
    :param initial_epoch: The initial epoch of the simulation. Defaults to 0.0.
    :param initial_state: The initial state of the simulation. In very few cases it will be optional.
    :param simulation_time: The total simulation time
    :param dependent_variables: The list of dependent variables. Usually not required.
    :return: The propagator settings.
    '''

    #  FIRST, THE CHECKS
    perform_propagator_checks(bodies, model, initial_epoch, initial_state, simulation_time, dependent_variables)

    # The bodies
    bodies_to_propagate = ['Phobos']
    central_bodies = ['Mars']

    # THE INTEGRATOR IS GOING TO BE THE SAME FOR ALL MODELS. LET'S JUST CREATE IT THE FIRST.
    time_step = 300.0  # These are 300s = 5min
    coefficients = propagation_setup.integrator.CoefficientSets.rkdp_87
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(time_step,
                                                                                      coefficients,
                                                                                      time_step,
                                                                                      time_step,
                                                                                      np.inf, np.inf)

    # THE TERMINATION CONDITION IS THE SAME FOR BOTH TRANSLATIONAL AND/OR ROTATIONAL DYNAMICS, AND HAS TO BE PASSED TO
    # THE MULTITYPE PROPAGATOR SETTINGS
    termination_condition = propagation_setup.propagator.time_termination(initial_epoch + simulation_time, True)

    # TRANSLATIONAL PROPAGATOR SETTINGS FOR MODELS S, A1, B AND C.
    if model in ['S', 'A1', 'B', 'C']:
        if model in ['B', 'C']: initial_translational_state = initial_state[:6]
        else: initial_translational_state = initial_state
        translational_propagator_settings = get_translational_dynamics_propagator(bodies,
                                                                                  bodies_to_propagate,
                                                                                  central_bodies,
                                                                                  initial_epoch,
                                                                                  initial_translational_state,
                                                                                  integrator_settings,
                                                                                  termination_condition,
                                                                                  dependent_variables)

        if model in ['S', 'A1']: settings_to_return = translational_propagator_settings

    # ROTATIONAL PROPAGATOR SETTINGS FOR MODELS A2, B AND C.
    if model in ['A2', 'B', 'C']:
        if model in ['B', 'C']: initial_rotational_state = initial_state[6:]
        else: initial_rotational_state = initial_state
        rotational_propagator_settings = get_rotational_dynamics_propagator(bodies,
                                                                            bodies_to_propagate,
                                                                            central_bodies,
                                                                            initial_epoch,
                                                                            initial_rotational_state,
                                                                            integrator_settings,
                                                                            termination_condition,
                                                                            dependent_variables)

        if model == 'A2': settings_to_return = rotational_propagator_settings

    # MULTI-TYPE DYNAMICS PROPAGATOR SETTINGS FOR MODELS B AND C
    if model in ['B', 'C']:
        propagator_list = [translational_propagator_settings, rotational_propagator_settings]
        combined_propagator_settings = propagation_setup.propagator.multitype(propagator_list,
                                                                              integrator_settings,
                                                                              initial_epoch,
                                                                              termination_condition,
                                                                              output_variables = dependent_variables)

        settings_to_return = combined_propagator_settings

    return settings_to_return


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                       POSTPROCESSING                                               ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def run_model_a1_checks(checks: list[int],
                        bodies: numerical_simulation.environment.SystemOfBodies,
                        simulator: numerical_simulation.SingleArcSimulator) -> None:

    if sum(checks) > 0:

        mars_mu = bodies.get('Mars').gravitational_parameter
        dependents = simulator.dependent_variable_history
        epochs_array = np.array(list(simulator.state_history.keys()))
        keplerian_history = extract_elements_from_history(simulator.dependent_variable_history, list(range(6, 12)))
        average_mean_motion, orbits = average_mean_motion_over_integer_number_of_orbits(keplerian_history, mars_mu)
        print('Average mean motion over', orbits, 'orbits:', average_mean_motion, 'rad/s =',
              average_mean_motion * 86400.0, 'rad/day')

    # Trajectory
    if checks[0]:
        trajectory_3d(simulator.state_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        plot_kepler_elements(keplerian_history)

    # Orbit is equatorial
    if checks[2]:
        sub_phobian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [13, 14]))
        sub_phobian_point[:, 1:] = bring_inside_bounds(sub_phobian_point[:, 1:], -PI, PI, include='upper')

        plt.figure()
        plt.scatter(sub_phobian_point[:, 2] * 360.0 / TWOPI, sub_phobian_point[:, 1] * 360.0 / TWOPI)
        plt.grid()
        plt.title('Sub-phobian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Sub-phobian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[3]:
        sub_martian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [4, 5]))
        sub_martian_point[:, 1:] = bring_inside_bounds(sub_martian_point[:, 1:], -PI, PI, include='upper')
        libration_history = extract_elements_from_history(simulator.dependent_variable_history, 5)
        libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)
        # phobos_mean_rotational_rate = 0.00022785759213999574  # In rad/s
        phobos_mean_rotational_rate = 0.000227995  # In rad/s

        plt.figure()
        plt.scatter(sub_martian_point[:, 2] * 360.0 / TWOPI, sub_martian_point[:, 1] * 360.0 / TWOPI)
        plt.grid()
        plt.title('Sub-martian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_martian_point[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_martian_point[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Sub-martian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker='.')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed',
                   c='r', label='Phobian mean motion')
        plt.title(r'Libration frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        plt.grid()
        # plt.xlim([0, 21])
        plt.legend()

    # Accelerations exerted by all third bodies. This will be used to assess whether the bodies are needed or not.
    if checks[4]:
        third_body_accelerations = result2array(extract_elements_from_history(dependents, list(range(15, 20))))
        third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
        plt.figure()
        for idx, body in enumerate(third_bodies):
            plt.semilogy(epochs_array / 86400.0, third_body_accelerations[:, idx + 1], label=body)
        plt.title('Third body accelerations')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel(r'Acceleration [m/s²]')
        plt.legend()
        plt.grid()
        
        
def run_model_a2_checks(checks: list[int],
                        bodies: numerical_simulation.environment.SystemOfBodies,
                        damping_results: numerical_simulation.propagation.DampedInitialRotationalStateResults,
                        check_undamped: bool) -> None:
    
    average_mean_motion = 0.0002278563609852602
    phobos_mean_rotational_rate = 0.000228035245
    
    if sum(checks) > 0:
        mars_mu = bodies.get('Mars').gravitational_parameter
        states_undamped = damping_results.forward_backward_states[0][0]
        dependents_undamped = damping_results.forward_backward_dependent_variables[0][0]
        states_damped = damping_results.forward_backward_states[-1][1]
        dependents_damped = damping_results.forward_backward_dependent_variables[-1][1]
        epochs_array = np.array(list(states_damped.keys()))

    # Trajectory
    if checks[0]:
        if check_undamped:
            cartesian_history_undamped = extract_elements_from_history(dependents_undamped, [15, 16, 17])
            trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
        cartesian_history = extract_elements_from_history(dependents_damped, [15, 16, 17])
        trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        if check_undamped:
            keplerian_history_undamped = extract_elements_from_history(dependents_undamped, [6, 7, 8, 9, 10, 11])
            plot_kepler_elements(keplerian_history_undamped, title='Undamped COEs')
        keplerian_history = extract_elements_from_history(dependents_damped, [6, 7, 8, 9, 10, 11])
        plot_kepler_elements(keplerian_history, title='Damped COEs')

    # Orbit is equatorial
    if checks[2]:
        if check_undamped:
            sub_phobian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [13, 14]))
            sub_phobian_point_undamped[:, 1:] = bring_inside_bounds(sub_phobian_point_undamped[:, 1:], -PI, PI,
                                                                    include='upper')
        sub_phobian_point = result2array(extract_elements_from_history(dependents_damped, [13, 14]))
        sub_phobian_point[:, 1:] = bring_inside_bounds(sub_phobian_point[:, 1:], -PI, PI, include='upper')

        if check_undamped:
            plt.figure()
            plt.scatter(sub_phobian_point_undamped[:, 2] * 360.0 / TWOPI,
                        sub_phobian_point_undamped[:, 1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(sub_phobian_point[:, 2] * 360.0 / TWOPI, sub_phobian_point[:, 1] * 360.0 / TWOPI,
                        label='Damped', marker='+')
            plt.grid()
            plt.title('Sub-phobian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            plt.title('Undamped sub-phobian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Damped sub-phobian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

    # Phobos' Euler angles. In a torque-free environment, the first two are constant and the third grows linearly as
    # indicated by the angular speed. This happens both in the undamped and damped cases. In an environment with torques,
    # the undamped angles contain free modes while the damped ones do not.
    if checks[3]:
        normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor,
                                                                       phobos_mean_rotational_rate)
        clean_signal = [TWOPI, 1]
        if check_undamped:
            euler_history_undamped = extract_elements_from_history(dependents_undamped, [0, 1, 2])
            euler_history_undamped = bring_history_inside_bounds(euler_history_undamped, 0.0, TWOPI)
            psi_freq_undamped, psi_amp_undamped = get_fourier_elements_from_history(
                extract_elements_from_history(euler_history_undamped, 0), clean_signal)
            theta_freq_undamped, theta_amp_undamped = get_fourier_elements_from_history(
                extract_elements_from_history(euler_history_undamped, 1), clean_signal)
            phi_freq_undamped, phi_amp_undamped = get_fourier_elements_from_history(
                extract_elements_from_history(euler_history_undamped, 2), clean_signal)
            euler_history_undamped = result2array(euler_history_undamped)
        euler_history = extract_elements_from_history(dependents_damped, [0, 1, 2])
        euler_history = bring_history_inside_bounds(euler_history, 0.0, TWOPI)
        psi_freq, psi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 0),
                                                              clean_signal)
        theta_freq, theta_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 1),
                                                                  clean_signal)
        phi_freq, phi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 2),
                                                              clean_signal)
        euler_history = result2array(euler_history)

        if check_undamped:
            plt.figure()
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:, 1] * 360.0 / TWOPI, label=r'$\psi$')
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:, 2] * 360.0 / TWOPI, label=r'$\theta$')
            # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
            plt.legend()
            plt.grid()
            # plt.xlim([0.0, 3.5])
            plt.title('Undamped Euler angles')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Angle [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, euler_history[:, 1] * 360.0 / TWOPI, label=r'$\psi$')
        plt.plot(epochs_array / 86400.0, euler_history[:, 2] * 360.0 / TWOPI, label=r'$\theta$')
        # plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
        plt.legend()
        plt.grid()
        # plt.xlim([0.0, 3.5])
        plt.title('Damped Euler angles')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Angle [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(psi_freq_undamped * 86400.0, psi_amp_undamped * 360 / TWOPI, label=r'$\psi$', marker='.')
            plt.loglog(theta_freq_undamped * 86400.0, theta_amp_undamped * 360 / TWOPI, label=r'$\theta$', marker='.')
            plt.loglog(phi_freq_undamped * 86400.0, phi_amp_undamped * 360 / TWOPI, label=r'$\phi$', marker='.')
            plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1),
                       ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                       label='Longitudinal normal mode')
            plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.title(r'Undamped frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            # plt.xlim([0, 70])
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(psi_freq * 86400.0, psi_amp * 360 / TWOPI, label=r'$\psi$', marker='.')
        plt.loglog(theta_freq * 86400.0, theta_amp * 360 / TWOPI, label=r'$\theta$', marker='.')
        plt.loglog(phi_freq * 86400.0, phi_amp * 360 / TWOPI, label=r'$\phi$', marker='.')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed',
                   c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                   label='Longitudinal normal mode')
        plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.title(r'Damped frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        # plt.xlim([0, 70])
        plt.grid()
        plt.legend()

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[4]:
        if check_undamped:
            sub_martian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [4, 5]))
            sub_martian_point_undamped[:, 1:] = bring_inside_bounds(sub_martian_point_undamped[:, 1:], -PI, PI,
                                                                    include='upper')
            libration_history_undamped = extract_elements_from_history(dependents_undamped, 5)
            libration_freq_undamped, libration_amp_undamped = get_fourier_elements_from_history(
                libration_history_undamped)
        sub_martian_point = result2array(extract_elements_from_history(dependents_damped, [4, 5]))
        sub_martian_point[:, 1:] = bring_inside_bounds(sub_martian_point[:, 1:], -PI, PI, include='upper')
        libration_history = extract_elements_from_history(dependents_damped, 5)
        libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)

        if check_undamped:
            plt.figure()
            plt.scatter(sub_martian_point_undamped[:, 2] * 360.0 / TWOPI,
                        sub_martian_point_undamped[:, 1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(sub_martian_point[:, 2] * 360.0 / TWOPI, sub_martian_point[:, 1] * 360.0 / TWOPI,
                        label='Damped', marker='+')
            plt.grid()
            plt.title(r'Sub-martian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            # plt.title('Undamped sub-martian point ($\omega = ' + str(phobos_mean_rotational_rate) + '$ rad/s)')
            plt.title('Undamped sub-martian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.scatter(sub_martian_point[:, 2] * 360.0 / TWOPI, sub_martian_point[:, 1] * 360.0 / TWOPI)
        plt.grid()
        plt.title(r'Damped sub-martian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_martian_point[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_martian_point[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Damped sub-martian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(libration_freq_undamped * 86400.0, libration_amp_undamped * 360 / TWOPI, marker='.')
            plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1),
                       ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                       label='Longitudinal normal mode')
            plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.title(r'Undamped libration frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker='.')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed',
                   c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                   label='Longitudinal normal mode')
        plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.title(r'Damped libration frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        plt.grid()
        plt.legend()

    # # Torques exerted by third bodies
    # if checks[5]:
    #
    #     # third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
    #     third_bodies = ['Sun', 'Earth', 'Mars', 'Deimos', 'Jupiter']
    #
    #     if check_undamped:
    #         third_body_torques_undamped = result2array(
    #             extract_elements_from_history(dependents_undamped, list(range(18, 23))))
    #         plt.figure()
    #         for idx, body in enumerate(third_bodies):
    #             plt.semilogy(epochs_array / 86400.0, third_body_torques_undamped[:, idx + 1], label=body)
    #         plt.title('Third body torques (undamped rotation)')
    #         plt.xlabel('Time [days since J2000]')
    #         plt.ylabel(r'Torque [N$\cdot$m]')
    #         plt.yticks([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
    #         plt.legend()
    #         plt.grid()
    #
    #     third_body_torques = result2array(extract_elements_from_history(dependents_damped, list(range(18, 23))))
    #     plt.figure()
    #     for idx, body in enumerate(third_bodies):
    #         plt.semilogy(epochs_array / 86400.0, third_body_torques[:, idx + 1], label=body)
    #     plt.title('Third body torques')
    #     plt.xlabel('Time [days since J2000]')
    #     plt.ylabel(r'Torque [N$\cdot$m]')
    #     plt.yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
    #     plt.legend()
    #     plt.grid()


def run_model_b_checks(checks: list[int],
                        bodies: numerical_simulation.environment.SystemOfBodies,
                        damping_results: numerical_simulation.propagation.DampedInitialRotationalStateResults,
                        check_undamped: bool) -> None:

    average_mean_motion = 0.0002278563609852602
    phobos_mean_rotational_rate = 0.000228035245

    if sum(checks) > 0:
        mars_mu = bodies.get('Mars').gravitational_parameter
        states_undamped = damping_results.forward_backward_states[0][0]
        dependents_undamped = damping_results.forward_backward_dependent_variables[0][0]
        states_damped = damping_results.forward_backward_states[-1][1]
        dependents_damped = damping_results.forward_backward_dependent_variables[-1][1]
        epochs_array = np.array(list(dependents_damped.keys()))

    # Trajectory
    if checks[0]:
        if check_undamped:
            cartesian_history_undamped = extract_elements_from_history(dependents_undamped, [15, 16, 17])
            trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
        cartesian_history = extract_elements_from_history(dependents_damped, [15, 16, 17])
        trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        if check_undamped:
            keplerian_history_undamped = extract_elements_from_history(dependents_undamped, [6, 7, 8, 9, 10, 11])
            plot_kepler_elements(keplerian_history_undamped, title='Undamped COEs')
        keplerian_history = extract_elements_from_history(dependents_damped, [6, 7, 8, 9, 10, 11])
        plot_kepler_elements(keplerian_history, title='Damped COEs')

    # Orbit is equatorial
    if checks[2]:
        if check_undamped:
            sub_phobian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [13, 14]))
            sub_phobian_point_undamped[:, 1:] = bring_inside_bounds(sub_phobian_point_undamped[:, 1:], -PI, PI,
                                                                    include='upper')
        sub_phobian_point = result2array(extract_elements_from_history(dependents_damped, [13, 14]))
        sub_phobian_point[:, 1:] = bring_inside_bounds(sub_phobian_point[:, 1:], -PI, PI, include='upper')

        if check_undamped:
            plt.figure()
            plt.scatter(sub_phobian_point_undamped[:, 2] * 360.0 / TWOPI,
                        sub_phobian_point_undamped[:, 1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(sub_phobian_point[:, 2] * 360.0 / TWOPI, sub_phobian_point[:, 1] * 360.0 / TWOPI,
                        label='Damped', marker='+')
            plt.grid()
            plt.title('Sub-phobian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            plt.title('Undamped sub-phobian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Damped sub-phobian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

    # Phobos' Euler angles.
    if checks[3]:
        normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor,
                                                                       phobos_mean_rotational_rate)
        clean_signal = [TWOPI, 1]
        if check_undamped:
            euler_history_undamped = extract_elements_from_history(dependents_undamped, [0, 1, 2])
            euler_history_undamped = bring_history_inside_bounds(euler_history_undamped, 0.0, TWOPI)
            psi_freq_undamped, psi_amp_undamped = get_fourier_elements_from_history(
                extract_elements_from_history(euler_history_undamped, 0), clean_signal)
            theta_freq_undamped, theta_amp_undamped = get_fourier_elements_from_history(
                extract_elements_from_history(euler_history_undamped, 1), clean_signal)
            phi_freq_undamped, phi_amp_undamped = get_fourier_elements_from_history(
                extract_elements_from_history(euler_history_undamped, 2), clean_signal)
            euler_history_undamped = result2array(euler_history_undamped)
        euler_history = extract_elements_from_history(dependents_damped, [0, 1, 2])
        euler_history = bring_history_inside_bounds(euler_history, 0.0, TWOPI)
        psi_freq, psi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 0),
                                                              clean_signal)
        theta_freq, theta_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 1),
                                                                  clean_signal)
        phi_freq, phi_amp = get_fourier_elements_from_history(extract_elements_from_history(euler_history, 2),
                                                              clean_signal)
        euler_history = result2array(euler_history)

        if check_undamped:
            plt.figure()
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:, 1] * 360.0 / TWOPI, label=r'$\psi$')
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:, 2] * 360.0 / TWOPI, label=r'$\theta$')
            # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
            plt.legend()
            plt.grid()
            # plt.xlim([0.0, 3.5])
            plt.title('Undamped Euler angles')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Angle [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, euler_history[:, 1] * 360.0 / TWOPI, label=r'$\psi$')
        plt.plot(epochs_array / 86400.0, euler_history[:, 2] * 360.0 / TWOPI, label=r'$\theta$')
        # plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
        plt.legend()
        plt.grid()
        # plt.xlim([0.0, 3.5])
        plt.title('Damped Euler angles')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Angle [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(psi_freq_undamped * 86400.0, psi_amp_undamped * 360 / TWOPI, label=r'$\psi$', marker='.')
            plt.loglog(theta_freq_undamped * 86400.0, theta_amp_undamped * 360 / TWOPI, label=r'$\theta$', marker='.')
            plt.loglog(phi_freq_undamped * 86400.0, phi_amp_undamped * 360 / TWOPI, label=r'$\phi$', marker='.')
            plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1),
                       ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                       label='Longitudinal normal mode')
            plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.title(r'Undamped frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            # plt.xlim([0, 70])
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(psi_freq * 86400.0, psi_amp * 360 / TWOPI, label=r'$\psi$', marker='.')
        plt.loglog(theta_freq * 86400.0, theta_amp * 360 / TWOPI, label=r'$\theta$', marker='.')
        plt.loglog(phi_freq * 86400.0, phi_amp * 360 / TWOPI, label=r'$\phi$', marker='.')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed',
                   c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                   label='Longitudinal normal mode')
        plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.title(r'Damped frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        # plt.xlim([0, 70])
        plt.grid()
        plt.legend()

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[4]:
        if check_undamped:
            sub_martian_point_undamped = result2array(extract_elements_from_history(dependents_undamped, [4, 5]))
            sub_martian_point_undamped[:, 1:] = bring_inside_bounds(sub_martian_point_undamped[:, 1:], -PI, PI,
                                                                    include='upper')
            libration_history_undamped = extract_elements_from_history(dependents_undamped, 5)
            p_history_undamped = extract_elements_from_history(dependents_undamped, 4)
            libration_freq_undamped, libration_amp_undamped = get_fourier_elements_from_history(
                libration_history_undamped)
            p_freq_undamped, p_amp_undamped = get_fourier_elements_from_history(p_history_undamped)
        sub_martian_point = result2array(extract_elements_from_history(dependents_damped, [4, 5]))
        sub_martian_point[:, 1:] = bring_inside_bounds(sub_martian_point[:, 1:], -PI, PI, include='upper')
        libration_history = extract_elements_from_history(dependents_damped, 5)
        p_history = extract_elements_from_history(dependents_damped, 4)
        libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)
        p_freq, p_amp = get_fourier_elements_from_history(p_history)

        if check_undamped:
            plt.figure()
            plt.scatter(sub_martian_point_undamped[:, 2] * 360.0 / TWOPI,
                        sub_martian_point_undamped[:, 1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(sub_martian_point[:, 2] * 360.0 / TWOPI, sub_martian_point[:, 1] * 360.0 / TWOPI,
                        label='Damped', marker='+')
            plt.grid()
            plt.title(r'Sub-martian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, sub_martian_point_undamped[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            # plt.title('Undamped sub-martian point ($\omega = ' + str(phobos_mean_rotational_rate) + '$ rad/s)')
            plt.title('Undamped sub-martian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.scatter(sub_martian_point[:, 2] * 360.0 / TWOPI, sub_martian_point[:, 1] * 360.0 / TWOPI)
        plt.grid()
        plt.title(r'Damped sub-martian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_martian_point[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_martian_point[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Damped sub-martian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(libration_freq_undamped * 86400.0, libration_amp_undamped * 360 / TWOPI, marker='.', label='Lon')
            plt.loglog(p_freq_undamped * 86400.0, p_amp_undamped * 360 / TWOPI, marker='.', label='Lat')
            plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1),
                       ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                       label='Longitudinal normal mode')
            plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed',
                       c='r')
            plt.title(r'Undamped libration frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker='.', label='Lon')
        plt.loglog(p_freq * 86400.0, p_amp * 360 / TWOPI, marker='.', label='Lat')
        plt.axline((phobos_mean_rotational_rate * 86400.0, 0), (phobos_mean_rotational_rate * 86400.0, 1), ls='dashed',
                   c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axline((normal_mode * 86400.0, 0), (normal_mode * 86400.0, 1), ls='dashed', c='k',
                   label='Longitudinal normal mode')
        plt.axline((2 * average_mean_motion * 86400.0, 0), (2 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.axline((3 * average_mean_motion * 86400.0, 0), (3 * average_mean_motion * 86400.0, 1), ls='dashed', c='r')
        plt.title(r'Damped libration frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        plt.grid()
        plt.legend()

    # # Torques exerted by third bodies
    # if checks[5]:
    #
    #     # third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
    #     third_bodies = ['Sun', 'Earth', 'Mars', 'Deimos', 'Jupiter']
    #
    #     if check_undamped:
    #         third_body_torques_undamped = result2array(
    #             extract_elements_from_history(dependents_undamped, list(range(18, 23))))
    #         plt.figure()
    #         for idx, body in enumerate(third_bodies):
    #             plt.semilogy(epochs_array / 86400.0, third_body_torques_undamped[:, idx + 1], label=body)
    #         plt.title('Third body torques (undamped rotation)')
    #         plt.xlabel('Time [days since J2000]')
    #         plt.ylabel(r'Torque [N$\cdot$m]')
    #         plt.yticks([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
    #         plt.legend()
    #         plt.grid()
    #
    #     third_body_torques = result2array(extract_elements_from_history(dependents_damped, list(range(18, 23))))
    #     plt.figure()
    #     for idx, body in enumerate(third_bodies):
    #         plt.semilogy(epochs_array / 86400.0, third_body_torques[:, idx + 1], label=body)
    #     plt.title('Third body torques')
    #     plt.xlabel('Time [days since J2000]')
    #     plt.ylabel(r'Torque [N$\cdot$m]')
    #     plt.yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
    #     plt.legend()
    #     plt.grid()


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


def compute_numerical_partials_in_state_transition_matrix(bodies: numerical_simulation.environment.SystemOfBodies,
                                                          propagator_settings: propagation_setup.propagator.PropagatorSettings,
                                                          perturbation_vector: np.ndarray) -> dict[float, np.ndarray] :

    original_initial_state = propagator_settings.initial_states

    # for idx in range(len(perturbation_vector)):
    for idx in [0]:

        current_perturbation = perturbation_vector[idx]
        diff = 2*current_perturbation

        # perturbation = np.zeros(len(perturbation_vector))
        # perturbation[idx] = current_perturbation
        # trajectory_plus = get_perturbed_trajectory(bodies,
        #                                            propagator_settings,
        #                                            perturbation)

        new_initial_state = original_initial_state.copy()
        new_initial_state[idx] = original_initial_state[idx] + current_perturbation
        propagator_settings.initial_states = new_initial_state
        # simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        # trajectory_plus = simulator.state_history


    state_transition_matrix_history = { 0.0 : np.eye(len(perturbation_vector)) }
    return state_transition_matrix_history


def get_perturbed_trajectory(bodies: numerical_simulation.environment.SystemOfBodies,
                             propagator_settings: propagation_setup.propagator.PropagatorSettings,
                             perturbation_vector: np.ndarray) -> dict[float, np.ndarray] :

    return


class MarsEquatorOfDate():

    def __init__(self, bodies: numerical_simulation.environment.SystemOfBodies):

        self.alpha_0 = np.radians(317.269202)
        self.delta_0 = np.radians(54.432516)
        self.W = np.radians(176.049863)

        self.phobos = bodies.get('Phobos')

        self.mars_to_j2000_rotation = self.get_mars_to_J2000_rotation_matrix()
        self.j2000_to_mars_rotation = self.mars_to_j2000_rotation.T

        return

    def get_mars_to_J2000_rotation_matrix(self) -> np.ndarray:

        psi = bring_inside_bounds(PI/2 + self.alpha_0, 0.0, TWOPI)
        theta = bring_inside_bounds(PI/2 - self.delta_0, 0.0, TWOPI)
        phi = bring_inside_bounds(self.W, 0.0, TWOPI)

        return euler_angles_to_rotation_matrix(np.array([psi, theta, phi]))

    def get_euler_angles_wrt_mars_equator(self) -> np.ndarray:

        phobos_to_J2000_rotation_matrix = self.phobos.body_fixed_to_inertial_frame
        phobos_to_mars_rotation_matrix = self.j2000_to_mars_rotation @ phobos_to_J2000_rotation_matrix

        return rotation_matrix_to_313_euler_angles(phobos_to_mars_rotation_matrix)

    def rotate_euler_angles_from_J2000_to_mars_equator(self, euler_angles_j2000: np.ndarray) -> np.ndarray:

        return rotate_euler_angles(euler_angles_j2000, self.get_mars_to_J2000_rotation_matrix())


def extract_estimation_output(estimation_output: numerical_simulation.estimation.EstimationOutput,
                              observation_times: list[float],
                              residual_type: str,
                              norm_position: bool = False) -> tuple:

    if residual_type not in ['position', 'orientation']:
        raise ValueError('(): Invalid residual type. Only "position" and "orientation" are allowed. Residual type provided is "' + residual_type + '".')

    number_of_iterations = estimation_output.residual_history.shape[1]
    iteration_array = list(range(number_of_iterations + 1))
    full_residual_history = np.hstack((estimation_output.residual_history, np.atleast_2d(estimation_output.final_residuals).T))
    if residual_type == 'position':
        residual_history = extract_position_residuals(full_residual_history,
                                                      observation_times,
                                                      norm_position)
        residual_rms_evolution = get_position_rms_evolution(residual_history, norm_position)
    if residual_type == 'orientation':
        residual_history = extract_orientation_residuals(estimation_output.residual_history,
                                                         observation_times,
                                                         number_of_iterations)
        residual_rms_evolution = get_orientation_rms_evolution(residual_history)
    parameter_evolution = dict(zip(iteration_array, estimation_output.parameter_history.T))

    return residual_history, parameter_evolution, residual_rms_evolution


def extract_orientation_residuals(residual_history: np.ndarray, observation_times: np.ndarray, number_of_iterations: float) -> dict:

    return


def get_orientation_rms_evolution(residual_history: dict) -> dict:

    return