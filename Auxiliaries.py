'''
THIS FILE CONTAINS FUNCTIONS THAT REQUIRE TUDAT. THE USE OF TUDAT IN LOGISTICS.PY AND ASTROMATH WAS LIMITED IF NOT
NON-EXISTENT. HERE, THE USE OF TUDAT IS EXTENSIVE. THIS FILE HAS A "LOGISTICS" SECTION AND AN "ASTROMATH" SECTION. THEY
GATHER FUNCTIONS WHICH COULD VERY WELL FALL IN EACH OF THE OTHER TWO FILES. HOWEVER, IT WAS DECIDED TO PUT THEM HERE DUE
TO THEIR REQUIRING TUDAT.
'''

# NATIVE IMPORTS
import sys
import os
import warnings
from datetime import datetime

from time import time
from cycler import cycler

if '/home/yorch/tudat-bundle/cmake-build-release/tudatpy' not in sys.path:
    sys.path.extend(['/home/yorch/tudat-bundle/cmake-build-release/tudatpy'])

from astromath import *

# TUDAT IMPORTS

# Domestic imports (i.e. only used in this file)
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries as mat2quat
from tudatpy.kernel.astro.element_conversion import quaternion_entries_to_rotation_matrix as quat2mat
from tudatpy.kernel.astro.element_conversion import cartesian_to_keplerian, true_to_mean_anomaly, semi_major_axis_to_mean_motion

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

# Public imports (i.e. required by all other scripts importing this module but not necessarily used here)
from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, estimation_setup
from tudatpy.util import compare_results
from tudatpy.plotting import trajectory_3d
from tudatpy.io import save2txt

import matplotlib.font_manager as fman

for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

fman.fontManager.addfont(r'/home/yorch/thesis/fonts/Gulliver Regular/Gulliver Regular.otf')

colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7f7f7f', '#bcbd22', '#17becf']
# plt.rc('font', family = 'Roboto Slab')
# plt.rc('font', family = 'Gulliver Regular')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')
plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
plt.rcParams['lines.markersize'] = 6.0

# color1, color2, color3, color4, color5 = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']

spice.load_standard_kernels([])
default_phobos_mean_rotational_rate = 0.000228035245

'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                          LOGISTICS                                                 ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def get_ephemeris_from_file(filename: str) -> environment_setup.ephemeris.EphemerisSettings:

    trajectory = read_vector_history_from_file(filename)
    imposed_trajectory = extract_elements_from_history(trajectory, [0, 1, 2, 3, 4, 5])
    ephemerides = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')

    return ephemerides


def acceleration_norm_from_body_on_phobos(body_exerting_acceleration: str) \
        -> propagation_setup.dependent_variable.SingleDependentVariableSaveSettings:

    point_mass = propagation_setup.acceleration.point_mass_gravity_type
    ret = propagation_setup.dependent_variable.single_acceleration_norm(point_mass, 'Phobos',
                                                                        body_exerting_acceleration)

    return ret


def torque_norm_from_body_on_phobos(body_exerting_torque: str) \
        -> propagation_setup.dependent_variable.SingleDependentVariableSaveSettings:

    point_mass = propagation_setup.torque.spherical_harmonic_gravitational_type
    ret = propagation_setup.dependent_variable.single_torque_norm(point_mass, 'Phobos', body_exerting_torque)

    return ret


def reduce_gravity_field(settings: numerical_simulation.environment_setup.gravity_field.GravityFieldSettings) -> None:

    new_sine_coefficients = np.zeros_like(settings.normalized_sine_coefficients)
    new_cosine_coefficients = np.zeros_like(settings.normalized_cosine_coefficients)

    new_cosine_coefficients[0, 0] = 1.0
    new_cosine_coefficients[2, 0] = settings.normalized_cosine_coefficients[2, 0]
    new_cosine_coefficients[2, 2] = settings.normalized_cosine_coefficients[2, 2]

    settings.normalized_sine_coefficients = new_sine_coefficients
    settings.normalized_cosine_coefficients = new_cosine_coefficients

    return


def save_initial_states(damping_results: numerical_simulation.propagation.RotationalProperModeDampingResults,
                        filename: str) -> None:

    initial_states_str = '\n'
    for iteration in range(len(damping_results.forward_backward_states)):
        initial_forward_state = list(damping_results.forward_backward_states[iteration][0].values())[0]
        initial_backward_state = list(damping_results.forward_backward_states[iteration][-1].values())[0]
        initial_states_str = initial_states_str + ' ' + str(iteration) + ' F ' + str(initial_forward_state) + '\n'
        initial_states_str = initial_states_str + ' ' + str(iteration) + ' B ' + str(initial_backward_state) + '\n'

    with open(filename, 'w') as file: file.write(initial_states_str)

    return


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                       BASIC ASTROMATH                                              ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def create_vector_interpolator(data: dict[float, np.ndarray]) -> interpolators.OneDimensionalInterpolatorVector:

    lagrange_settings = interpolators.lagrange_interpolation(number_of_points = 8)
    interpolator = interpolators.create_one_dimensional_vector_interpolator(data, lagrange_settings)

    return interpolator


def cartesian_to_keplerian_history(cartesian_history: dict,
                                   gravitational_parameter: float) -> dict:

    epochs_list = list(cartesian_history.keys())
    keplerian_history = dict.fromkeys(epochs_list)
    for key in epochs_list:
        keplerian_history[key] = cartesian_to_keplerian(cartesian_history[key], gravitational_parameter)

    return keplerian_history


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


def quaternion_entries_to_euler_angles(quaternion: np.ndarray[4]) -> np.ndarray[3]:

    rotation_matrix = quat2mat(quaternion)
    theta = np.arccos(rotation_matrix[2,2])
    psi = np.arctan2(rotation_matrix[0,2], -rotation_matrix[1,2])
    phi = np.arctan2(rotation_matrix[2,0], rotation_matrix[2,1])

    return np.array([psi, theta, phi])


def quaternion_to_euler_history(quaternion_history: dict) -> dict:

    epochs = list(quaternion_history.keys())
    values = [quaternion_entries_to_euler_angles(quaternion) for quaternion in list(quaternion_history.values())]

    return dict(zip(epochs, values))


def quaternion_to_matrix_history(quaternion_history: dict) -> dict:

    epochs = list(quaternion_history.keys())
    values = [quat2mat(quaternion) for quaternion in list(quaternion_history.values())]

    return dict(zip(epochs, values))


class MarsEquatorOfDate:

    def __init__(self, bodies: numerical_simulation.environment.SystemOfBodies):

        self.alpha_0 = np.radians(317.269202)
        self.delta_0 = np.radians(54.432516)
        self.W = np.radians(176.049863)

        self.phobos = bodies.get('Phobos')
        self.mars = bodies.get('Mars')

        self.mars_to_j2000_rotation = self.get_mars_to_J2000_rotation_matrix()
        self.j2000_to_mars_rotation = self.mars_to_j2000_rotation.T

        return

    def get_mars_to_J2000_rotation_matrix(self) -> np.ndarray:

        psi = bring_inside_bounds(PI/2 + self.alpha_0, 0.0, TWOPI)
        theta = bring_inside_bounds(PI/2 - self.delta_0, 0.0, TWOPI)
        phi = bring_inside_bounds(self.W, 0.0, TWOPI)

        return euler_angles_to_rotation_matrix(np.array([psi, theta, phi]))

    def keplerian_state_in_mars_reference_frame(self) -> np.ndarray:

        keplerian_state_in_earth_equator = cartesian_to_keplerian(self.phobos.state, self.mars.gravitational_parameter)
        orbit_angles_in_j2000 = np.array([keplerian_state_in_earth_equator[4], # RAAN
                                          keplerian_state_in_earth_equator[2], # Inclination
                                          keplerian_state_in_earth_equator[3]]) # Argument of periapsis
        orbit_angles_in_mars_equator = self.rotate_euler_angles_from_J2000_to_mars_equator(orbit_angles_in_j2000)
        keplerian_state_in_mars_equator = keplerian_state_in_earth_equator.copy()
        keplerian_state_in_mars_equator[2:5] = orbit_angles_in_mars_equator

        return keplerian_state_in_mars_equator

    def get_euler_angles_wrt_mars_equator(self) -> np.ndarray:

        phobos_to_J2000_rotation_matrix = self.phobos.body_fixed_to_inertial_frame
        phobos_to_mars_rotation_matrix = self.j2000_to_mars_rotation @ phobos_to_J2000_rotation_matrix

        return rotation_matrix_to_313_euler_angles(phobos_to_mars_rotation_matrix)

    def rotate_euler_angles_from_J2000_to_mars_equator(self, euler_angles_j2000: np.ndarray) -> np.ndarray:

        return rotate_euler_angles(euler_angles_j2000, self.get_mars_to_J2000_rotation_matrix())


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                 ENVIRONMENT INFRASTRUCTURE                                         ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def get_solar_system(model: str,
                     translational_ephemeris_file: str = '',
                     rotational_ephemeris_file: str = '',
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
    "scaled_libration_amplitude".

    :param model: The model to be used. It can be S, A1, A2, B or C.
    :param translational_ephemeris_file: The name of a file containing a history of state vectors to assign to Phobos.
    :param rotational_ephemeris_file: The name of a file containing a history of state vectors to assign to Phobos.
    :param **additional_inputs: Other stuff. For now, the only thing supported is "libration_amplitude".
    :return: bodies
    '''

    if model not in ['S', 'A1', 'A2', 'B', 'C']:
        raise ValueError('Model provided is invalid.')

    if model == 'A2' and translational_ephemeris_file == '':
        warnings.warn('The model you selected requires a translational ephemeris. No ephemeris file was provided. Reverting to default spice ephemeris.')

    # WE FIRST CREATE MARS.
    bodies_to_create = ["Sun", "Earth", "Mars", "Deimos", "Jupiter"]
    global_frame_origin = "Mars"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

    # WE THEN CREATE PHOBOS USING THE INPUTS.
    body_settings.add_empty_settings('Phobos')
    # Ephemeris model. If no file is provided, defaults to spice ephemeris.
    if translational_ephemeris_file == '':
        body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
    else:
        imposed_trajectory = read_vector_history_from_file(translational_ephemeris_file)
        body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.tabulated(imposed_trajectory, 'Mars', 'J2000')
    # Rotational model. If no file is provided, defaults to synchronous rotation.
    if rotational_ephemeris_file == '':
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
    if model == 'A1' and rotational_ephemeris_file == '':
        if 'scaled_libration_amplitude' in additional_inputs:
            scaled_amplitude = additional_inputs['scaled_libration_amplitude']
        elif 'model_for_libration_amplitude_computation' in additional_inputs:
            temp = additional_inputs['model_for_libration_amplitude_computation'].lower()
            dependents = read_vector_history_from_file(os.getcwd() + '/ephemeris/associated-dependents/' + temp + '.dat')
            scaled_libration_amplitude = compute_scaled_libration_amplitude_from_dependent_variables(dependents)
            # scaled_amplitude = 2.695220284671387  # Directly computed in a numerical way from model B. (RKF10(12) dt = 5min)
            # scaled_amplitude = 2.6952203863816266 # Directly computed in a numerical way from model B. (RKDP7(8) dt = 4.5min)
        bodies.get('Phobos').rotation_model.libration_calculator = numerical_simulation.environment.DirectLongitudeLibrationCalculator(scaled_amplitude)

    return bodies


def get_gravitational_field(frame_name: str,
                            field_type: str = 'QUAD') -> environment_setup.gravity_field.GravityFieldSettings:

    # The gravitational field implemented here is that in Le Maistre et al. (2019).

    datadir = os.getcwd() + '/Normalized gravity fields/'
    cosines_file = datadir + 'cosines Le Maistre.txt'
    sines_file = datadir + 'sines Le Maistre.txt'

    phobos_gravitational_parameter = 1.06e16*constants.GRAVITATIONAL_CONSTANT
    phobos_reference_radius = 14e3

    phobos_normalized_cosine_coefficients = read_matrix_from_file(cosines_file, [5,5])
    phobos_normalized_sine_coefficients = read_matrix_from_file(sines_file, [5,5])

    settings_to_return = environment_setup.gravity_field.spherical_harmonic(
        phobos_gravitational_parameter,
        phobos_reference_radius,
        phobos_normalized_cosine_coefficients,
        phobos_normalized_sine_coefficients,
        associated_reference_frame = frame_name)

    if field_type == 'QUAD': reduce_gravity_field(settings_to_return)

    return settings_to_return


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                 INTEGRATION INFRASTRUCTURE                                         ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


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
##########                                 PROPAGATION INFRASTRUCTURE                                         ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def get_propagator_settings(model: str,
                            bodies: numerical_simulation.environment.SystemOfBodies,
                            initial_epoch: float,
                            initial_state: np.ndarray,
                            simulation_time: float,
                            dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = [],
                            **optional_settings)\
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
    :param integrator_time_step:
    :return: The propagator settings.
    '''

    #  FIRST, THE CHECKS
    perform_propagator_checks(bodies, model, initial_epoch, initial_state, simulation_time, dependent_variables)

    # The bodies
    bodies_to_propagate = ['Phobos']
    central_bodies = ['Mars']

    # THE INTEGRATOR IS GOING TO BE THE SAME FOR ALL MODELS. LET'S JUST CREATE IT THE FIRST.
    if 'integrator_settings' in optional_settings:
        integrator_settings = optional_settings['integrator_settings']
    else:
        if 'time_step' in optional_settings:
            time_step = optional_settings['time_step']
        else:
            time_step = 300.0  # These are 300s = 5min
            # time_step = 270.0  # These are 270s = 4.5min
            # time_step = 210.0  # These are 210s = 3.5min
        coefficients = propagation_setup.integrator.CoefficientSets.rkf_108
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


def perform_propagator_checks(bodies: numerical_simulation.environment.SystemOfBodies,
                              model: str,
                              initial_epoch: float,
                              initial_state: np.ndarray,
                              simulation_time: float,
                              dependent_variables: list[propagation_setup.dependent_variable.PropagationDependentVariables] = []) \
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
    quad_gravity = sum(sum(abs(bodies.get('Phobos').gravity_field_model.sine_coefficients))) == 0.0
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
        equator = MarsEquatorOfDate(bodies)
        euler_angles_wrt_mars_equator_dependent_variable = \
            propagation_setup.dependent_variable.custom_dependent_variable(equator.get_euler_angles_wrt_mars_equator, 3)
        keplerian_state_wrt_mars = \
            propagation_setup.dependent_variable.custom_dependent_variable(equator.keplerian_state_in_mars_reference_frame, 6)
        dependent_variables_to_save = [ euler_angles_wrt_mars_equator_dependent_variable,  # 0, 1, 2
                                        propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                        keplerian_state_wrt_mars,  # 6, 7, 8, 9, 10, 11
                                        propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                        acceleration_norm_from_body_on_phobos('Sun'), # 15
                                        acceleration_norm_from_body_on_phobos('Earth'),  # 16
                                        mars_acceleration_dependent_variable,  # 17
                                        acceleration_norm_from_body_on_phobos('Deimos'),  # 18
                                        acceleration_norm_from_body_on_phobos('Jupiter')  # 19
                                        ]

    elif model in ['A2', 'B', 'C']:
        equator = MarsEquatorOfDate(bodies)
        euler_angles_wrt_mars_equator_dependent_variable = \
            propagation_setup.dependent_variable.custom_dependent_variable(equator.get_euler_angles_wrt_mars_equator, 3)
        keplerian_state_wrt_mars = \
            propagation_setup.dependent_variable.custom_dependent_variable(equator.keplerian_state_in_mars_reference_frame, 6)
        dependent_variables_to_save = [euler_angles_wrt_mars_equator_dependent_variable,  # 0, 1, 2
                                       propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                       keplerian_state_wrt_mars,  # 6, 7, 8, 9, 10, 11
                                       propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                       propagation_setup.dependent_variable.relative_position('Phobos', 'Mars')  # 15, 16, 17
                                       ]

    else: dependent_variables_to_save = []

    return dependent_variables_to_save


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                 ESTIMATION INFRASTRUCTURE                                          ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def print_estimated_parameters(estimated_parameters: list[str]) -> str:

    parameters_str = ''

    if 'initial state' in estimated_parameters:
        parameters_str = parameters_str + '\t- Initial state\n'
    if 'A' in estimated_parameters:
        parameters_str = parameters_str + '\t- Libration amplitude\n'
    if 'C20' in estimated_parameters and 'C22' in estimated_parameters:
        parameters_str = parameters_str + '\t- C20\n'
        parameters_str = parameters_str + '\t- C22\n'
    elif 'C20' in estimated_parameters:
        parameters_str = parameters_str + '\t- C20\n'
    elif 'C22' in estimated_parameters:
        parameters_str = parameters_str + '\t- C22\n'

    print('\nParameters to be estimated:\n' + parameters_str)

    return parameters_str


def get_parameter_set(estimated_parameters: list[str],
                      bodies: numerical_simulation.environment.SystemOfBodies,
                      propagator_settings: propagation_setup.propagator.PropagatorSettings | None = None,
                      return_only_settings_list: bool = True) -> tuple:

    parameter_settings = []

    if propagator_settings is not None:
        parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    else:
        raise ValueError('(get_parameter_set): Propagator settings required but not provided.')

    if 'A' in estimated_parameters:
        parameter_settings = parameter_settings + [estimation_setup.parameter.scaled_longitude_libration_amplitude('Phobos')]
    if 'C20' in estimated_parameters and 'C22' in estimated_parameters:
        parameter_settings = parameter_settings + [
            estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 0), (2, 2)])]
    elif 'C20' in estimated_parameters:
        parameter_settings = parameter_settings + [
            estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 0)])]
    elif 'C22' in estimated_parameters:
        parameter_settings = parameter_settings + [
            estimation_setup.parameter.spherical_harmonics_c_coefficients_block('Phobos', [(2, 2)])]

    if return_only_settings_list:
        return parameter_settings
    else:
        parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
        return parameters_to_estimate


def get_observation_history(observation_times: list[float],
                            observation_collection: numerical_simulation.estimation.ObservationCollection) -> dict:

    observation_history = dict.fromkeys(observation_times)
    for idx, epoch in enumerate(observation_times):
        observation_history[epoch] = observation_collection.concatenated_observations[3*idx:3*(idx+1)]

    return observation_history


def convert_residuals_to_rsw(cartesian_residual_history: dict,
                             ephemeris_states_at_observation_times: dict[float, np.ndarray]) -> dict:

    epochs = list(cartesian_residual_history.keys())
    residuals_in_rsw = dict.fromkeys(epochs)
    number_of_iterations = int(len(cartesian_residual_history[epochs[0]]) / 3)

    # for iteration in range(number_of_iterations):
    #     # AQUI EL PROBLEMA ES QUE LOS STATES DE LA ULTIMISIMA ITERACION (CORRESPONDIENTES A LOS ESTIMATED PARAMETERS FINALES) NO ESTÁN DENTRO DEL STATE HISTORY PER ITERATION.
    #     for epoch in epochs:
    #         if iteration == 0:
    #             residuals_in_rsw[epoch] = np.zeros(3 * number_of_iterations)
    #         R = inertial_to_rsw_rotation_matrix(states.interpolate(epoch))
    #         residuals_in_rsw[epoch][3 * iteration:3 * (iteration + 1)] = R @ cartesian_residual_history[epoch][
    #                                                                          3 * iteration:3 * (iteration + 1)]
    #

    for epoch in epochs:
        residuals_in_rsw[epoch] = np.zeros(3*number_of_iterations)
        R = inertial_to_rsw_rotation_matrix(ephemeris_states_at_observation_times[epoch])
        for iteration in range(number_of_iterations):
            idx1, idx2 = int(3*iteration), int(3*(iteration+1))
            residuals_in_rsw[epoch][idx1:idx2] = R @ cartesian_residual_history[epoch][idx1:idx2]

    return residuals_in_rsw


class EstimationSettings:

    def __init__(self, settings_file: str):

        self.source_file = settings_file
        self.estimation_settings = dict()
        self.observation_settings = dict()
        self.ls_convergence_settings = dict()
        self.postprocess_settings = dict()
        self.test_functionalities = dict()
        self.execution_settings = dict()

        self.read_settings_from_file()
        self.run_diagnostics_on_settings()

        self.number_of_estimations = len(self.estimation_settings['duration of estimated arc'])

        return

    def read_settings_from_file(self) -> None:

        offset_last = False  # Leave this as is. You will see the point of it down below.

        with open(self.source_file, 'r') as file: all_lines = [line for line in file.readlines() if line != '\n']

        dicts = [self.estimation_settings, self.observation_settings, self.ls_convergence_settings,
                 self.postprocess_settings, self.test_functionalities, self.execution_settings]
        dict_idx = -1
        for line in all_lines:
            if '#####' in line:
                continue
            if '####' in line:
                dict_idx = dict_idx + 1
                continue
            if ':' in line:

                dict_key = line.split(':')[0].removeprefix('· ').lower()
                dict_value = line.split('#')[0]  # Take everything that is NOT a comment
                while dict_value[-1] == ' ' or dict_value[-1] == '\t': dict_value = dict_value[:-1]  # Remove all trailing spaces or tabs
                dict_value = dict_value.split(':')[1].removesuffix('\n').replace('\t', '')  # Take the value itself (everything after the colon).
                while dict_value[0] == ' ' or dict_value[0] == '\t': dict_value = dict_value[1:]  # Remove all preceding spaces or tabs

                # A SPECIAL CASE HERE IS THAT THERE IS A UNIT IN THE INPUT (MINUTE, DAY, YEAR, ...)
                if 'YEAR' in dict_value or 'DAY' in dict_value or 'HOUR' in dict_value or 'MINUTE' in dict_value or \
                    'SECOND' in dict_value:

                    # There are four settings that enter this "if" clause:
                    #   ESTIMATION INITIAL EPOCH
                    #   DURATION OF ESTIMATED ARC
                    #   EPOCH OF FIRST OBSERVATION
                    #   EPOCH OF LAST OBSERVATION
                    # Usually, these are a scalar followed by the unit. But DURATION OF ESTIMATED ARC might be a sequence of scalars.
                    # Thus, the following try-except will discriminate this particular case.
                    unit = dict_value.split(' ')[-1]  # This is the unit itself
                    if 'OFFSET' in unit:
                        unit = dict_value.split(' ')[-2]  # This is the unit itself
                    try:
                        value = float(dict_value.removeprefix(' ').split(' ')[0])  # This is the case where it is a float.
                    except:
                        value = np.array([float(element) for element in dict_value.removesuffix(' ' + unit).split(', ')]) # This is an array with all values.

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
                            offset_last = True
                    else:
                        dicts[dict_idx][dict_key] = value

                    # The work with this particular line is done. We can continue directly to the next.
                    continue

                # THE REST OF THE CASES IS FLOATS, STRINGS OR SEQUENCES THEREOF SEPARATED BY ", "
                try:
                    dict_value = float(dict_value)  # We try and see if it can be converted into a float.
                except:
                    if ',' not in dict_value:  # In this case, the value is not a float but also not a sequence, therefore just a string.
                        if dict_value == 'True' or dict_value == 'False':  # If it represents a boolean, it should be converted.
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

        if type(self.estimation_settings['estimated parameters']) != list:
            self.estimation_settings['estimated parameters'] = [self.estimation_settings['estimated parameters']]

        if type(self.estimation_settings['duration of estimated arc']) == float:
            self.estimation_settings['duration of estimated arc'] = [self.estimation_settings['duration of estimated arc']]
            if offset_last:
                self.observation_settings['epoch of last observation'] = [self.observation_settings['epoch of last observation']]
        else:
            self.estimation_settings['duration of estimated arc'] = list(self.estimation_settings['duration of estimated arc'])
            if offset_last:
                self.observation_settings['epoch of last observation'] = list(self.observation_settings['epoch of last observation'])

        return

    def run_diagnostics_on_settings(self) -> None:

        if self.estimation_settings['estimation model'] not in ['S', 'A1', 'A2', 'B', 'C']:
            raise ValueError('Invalid estimation model.')

        if self.observation_settings['observation model'] in ['S', 'A1']:
            raise ValueError('The observation model you have selected is not supposed to be used as such. The program '
                             'has safely stopped now before shit breaks in mores spectacular ways.')

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
        tf_e = self.estimation_settings['initial estimation epoch'] + np.array(self.estimation_settings['duration of estimated arc'])
        t0_o = self.observation_settings['epoch of first observation']
        tf_o = np.array(self.observation_settings['epoch of last observation'])
        params = self.estimation_settings['estimated parameters']

        if sum(tf_o <= t0_o):
            raise ValueError('(Some of) the epoch(s) of the last observation is/are lower than (or equal to) the epoch of the first '
                             'observation.')

        if sum(tf_e <= tf_o):
            warnings.warn('(Some of) your initial estimation epoch(s) is greater than (or equal to) your final estimation epoch. '
                          'This will cause the propagation of your equations of motion and variational equations to run'
                          ' backwards. The "initial state" will be that at the final epoch.')

        if t0_o < t0_e or sum(tf_o > tf_e):
            warnings.warn('Some of your observations are outside your estimation period. They will be ignored.')

        if 'A' in params and 'C20' in params: ill_posedness = True
        elif 'A' in params and 'C22' in params: ill_posedness = True
        elif 'C20' in params and 'C22' in params: ill_posedness = True
        elif 'A' in params and 'C20' in params and 'C22' in params: ill_posedness = True
        else: ill_posedness = False
        if ill_posedness:
            warnings.warn('Your estimated parameters include combinations that are known to be prone to ill-posedness of'
                          ' the least squares problem. Expect large condition numbers.')

        if not self.test_functionalities['test mode'] and self.estimation_settings['estimation model'] == self.observation_settings['observation model']:
            warnings.warn('Test mode was not selected, yet the observation and estimation models are identical. First '
                          'estimation iteration already departs from truth.')

        if self.postprocess_settings['plot normed residuals'] and not self.estimation_settings['norm position residuals']:
            warnings.warn('Plots for normed residuals were requested but residuals were not normed. The former will be ignored.')

        if self.postprocess_settings['plot rsw residuals'] and not self.estimation_settings['convert residuals to rsw']:
            warnings.warn('Plots for RSW residuals were requested but residuals were not converted. The former will be ignored.')

        return

    def batch_to_single_run(self, duration_idx: int) -> list[str]:

        '''

        This function takes a settings file that was intended for an estimation batch (i.e. it had a list in the
        "duration of estimated arc" entry) and converts it into a settings file for a single run.

        :param duration_idx:
        :return:
        '''

        with open(self.source_file, 'r') as file:
            file_lines = file.readlines()

        estimation_duration = self.estimation_settings['duration of estimated arc'][duration_idx]
        if estimation_duration % 86400.0 == 0.0:
            estimation_duration = estimation_duration / 86400.0
            day = True
        else:
            day = False

        for idx, line in enumerate(file_lines):
            if 'DURATION OF ESTIMATED ARC' in line:
                things = line.split(':\t')
                things[1] = str(estimation_duration)
                if day:
                    things[1] = things[1] + ' DAY\n'
                file_lines[idx] = ':\t'.join(things)

        return file_lines

    def get_length_of_estimated_state(self) -> int:

        if self.estimation_settings['estimation model'] in ['S', 'A1']:
            length_of_estimated_state = 6
        elif self.estimation_settings['estimation model'] == 'A2':
            length_of_estimated_state = 7
        else:
            length_of_estimated_state = 13

        return length_of_estimated_state

    def get_observation_times(self, estimation_idx: int) -> list[float]:

        N = int((self.observation_settings['epoch of last observation'][estimation_idx] - \
                 self.observation_settings['epoch of first observation']) / \
                self.observation_settings['observation frequency']) + 1

        observation_times = np.linspace(self.observation_settings['epoch of first observation'],
                                        self.observation_settings['epoch of last observation'][estimation_idx],
                                        N)

        return observation_times

    def get_full_state_at_observation_times(self, estimation_idx: int) -> dict[float, np.ndarray]:

        state_history_at_observation_times = dict.fromkeys(self.get_observation_times(estimation_idx))

        model = self.observation_settings['observation model'][0].lower()
        ephemeris_file = 'ephemeris/translation-' + model + '.eph'
        state_history = create_vector_interpolator(read_vector_history_from_file(ephemeris_file))

        for epoch in self.get_observation_times(estimation_idx):
            state_history_at_observation_times[epoch] = state_history.interpolate(epoch)

        return state_history_at_observation_times

    def get_estimation_type(self) -> str | None:

        estimation_type = ''
        obs = self.observation_settings['observation type']
        params = self.estimation_settings['estimated parameters']
        model = self.estimation_settings['estimation model']
        if model == 'A1':
            if params == ['initial state', 'C20']:
                estimation_type = 'alpha-1'
            elif params == ['initial state', 'C22']:
                estimation_type = 'alpha-2'
            elif params == ['initial state', 'C20', 'C22']:
                estimation_type = 'alpha'
            elif params == ['initial state', 'A']:
                estimation_type = 'bravo'
            elif params == ['initial state', 'A', 'C20', 'C22']:
                estimation_type = 'charlie'
            else: pass

        if model == 'B' and obs == 'position':
            estimation_type = 'delta'
            if params == ['initial state', 'C20']:
                estimation_type = estimation_type + '-1'
            elif params == ['initial state', 'C22']:
                estimation_type = estimation_type + '-2'

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

    def get_initial_rotational_state(self, type: str, model: str = '') -> np.ndarray:

        if model == '':
            if type == 'estimation':
                model = self.estimation_settings['estimation model']
            else:
                model = self.observation_settings['observation model']

        if model in ['S', 'A1']:
            raise ValueError('You\'re in the wrong place my dude')

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

    def get_plot_legends(self) -> list[str]:

        legends = []
        parameter_list = self.estimation_settings['estimated parameters']
        if type(parameter_list) != list:
            parameter_list = [parameter_list]
        for parameter in parameter_list:
            if parameter == 'initial state':
                if self.estimation_settings['estimation model'] == 'A2':
                    translational_state_legends = []
                else:
                    translational_state_legends = [r'$x_o$', r'$y_o$', r'$z_o$', r'$v_{x,o}$', r'$v_{y,o}$', r'$v_{z,o}$']
                if self.estimation_settings['estimation model'] in ['A2', 'B', 'C']:
                    rotational_state_legends = [r'$q_{0,o}$', r'$q_{1,o}$', r'$q_{2,o}$', r'$q_{3,o}$', r'$\omega_{1,o}$', r'$\omega_{2,o}$', r'$\omega_{3,o}$']
                else:
                    rotational_state_legends = []
                legends = legends + translational_state_legends + rotational_state_legends
            elif parameter == 'A':
                legends = legends + [r'$A$']
            elif parameter == 'C20':
                legends = legends + [r'$C_{2,0}$']
            elif parameter == 'C22':
                legends = legends + [r'$C_{2,2}$']
            else:
                raise ValueError('(EstimationSettings.get_plot_legends): Invalid parameter string.')

        return legends

    def get_true_parameters(self) -> np.ndarray:

        true_parameters = []
        parameter_list = self.estimation_settings['estimated parameters']
        if type(parameter_list) != list:
            parameter_list = [parameter_list]
        for parameter in parameter_list:
            if parameter == 'initial state':
                true_initial_state = self.get_initial_state('true')
                if self.estimation_settings['estimation model'] in ['S', 'A1']:
                    true_initial_state = true_initial_state[:6]
                if self.estimation_settings['estimation model'] == 'A2':
                    true_initial_state = true_initial_state[6:]
                true_parameters = true_parameters + list(true_initial_state)
            elif parameter == 'A':
                true_parameters = true_parameters + [2.695220284671387]
            elif parameter == 'C20':
                true_parameters = true_parameters + [-0.029243]
            elif parameter == 'C22':
                true_parameters = true_parameters + [0.015664]
            else:
                raise ValueError('(EstimationSettings.get_true_parameters): Invalid parameter string.')

        return np.array(true_parameters)

    def scaled_to_tidal_libration_ampltiude(self, parameter_evolution: np.ndarray) -> np.ndarray:

        # WARNING: Index 0 of parameter_evolution is the iteration number! That's why we have index[:,index+1] rather
        # than just new[:,index] below.

        legends = self.get_plot_legends()
        new = parameter_evolution
        if r'$A$' in legends:
            index = legends.index(r'$A$')
            dependents_file = os.getcwd() + '/ephemeris/associated-dependents/' + self.observation_settings['observation model'].lower() + '.dat'
            dependents = read_vector_history_from_file(dependents_file)
            eccentricity = compute_eccentricity_from_dependent_variables(dependents)
            new[:,index+1] = new[:,index+1] * eccentricity

        return new

    def get_covariance_dimensions(self) -> list[int]:
        dim = 0
        params = self.estimation_settings['estimated parameters']
        if type(params) != list:
            params = [params]
        for param in params:
            if param == 'initial state':
                dim = dim + 6
            else:
                dim = dim + 1

        return [int(dim), int(dim)]


def extract_estimation_output(estimation_output: numerical_simulation.estimation.EstimationOutput,
                              estimation_settings: EstimationSettings,
                              estimation_idx: int) -> tuple:

    observation_times = estimation_settings.get_observation_times(estimation_idx)
    residual_type = estimation_settings.observation_settings['observation type']
    norm_position = estimation_settings.estimation_settings['norm position residuals']
    residuals_to_rsw = estimation_settings.estimation_settings['convert residuals to rsw']

    ephemeris_states_at_observation_times = estimation_settings.get_full_state_at_observation_times(estimation_idx)

    if residual_type not in ['position', 'orientation']:
        raise ValueError('(extract_estimation_output): Invalid residual type. Only "position" and "orientation" are '
                         'allowed. Residual type provided is "' + residual_type + '".')

    number_of_iterations = estimation_output.residual_history.shape[1]
    iteration_array = list(range(number_of_iterations + 1))
    full_residual_history = np.hstack((estimation_output.residual_history, np.atleast_2d(estimation_output.final_residuals).T))
    residual_histories = [rearrange_position_residuals(full_residual_history, ephemeris_states_at_observation_times)]
    if norm_position:
        residual_histories.append(norm_position_residuals(residual_histories[0]))
    if residuals_to_rsw:
        residual_histories.append(convert_residuals_to_rsw(residual_histories[0],
                                                               ephemeris_states_at_observation_times))
    residual_statistical_indicators = get_position_statistical_indicators_evolution(
        residual_histories,
        estimation_settings.estimation_settings['norm position residuals'],
        estimation_settings.estimation_settings['convert residuals to rsw']
    )

    parameter_evolution = dict(zip(iteration_array, estimation_output.parameter_history.T))

    return residual_histories, parameter_evolution, residual_statistical_indicators


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
        dependents = dict2array(simulator.dependent_variable_history)
        # epochs_array = np.array(list(simulator.state_history.keys()))
        # keplerian_history = extract_elements_from_history(simulator.dependent_variable_history, list(range(6, 12)))
        keplerian_history = dependents[:,[0, 6, 7, 8, 9, 10, 11]]
        mean_motion_history = keplerian_history[:,[0,1]]
        mean_motion_history[:,1] = np.sqrt(mars_mu / mean_motion_history[:,1] ** 3)
        average_mean_motion, orbits = average_over_integer_number_of_orbits(mean_motion_history, keplerian_history)
        print('Average mean motion over', orbits, 'orbits:', average_mean_motion, 'rad/s =',
              average_mean_motion * 86400.0, 'rad/day')

        epochs = dependents[:,0]

    # Trajectory
    if checks[0]:
        trajectory_3d(simulator.state_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        plot_kepler_elements(keplerian_history)

    # Orbit is equatorial
    if checks[2]:
        sub_phobian_point = dependents[:,[0,14,15]]
        sub_phobian_point[:, 1:] = bring_inside_bounds(sub_phobian_point[:, 1:], -PI, PI, include='upper')

        plt.figure()
        plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:,1] * 360.0 / TWOPI)
        plt.grid()
        plt.title('Sub-phobian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Sub-phobian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[3]:

        librations = dependents[:,[0,5,6]]
        librations[:,1:] = bring_inside_bounds(librations[:,1:], -PI, PI, include='upper')
        librations_fourier = fourier_transform(librations, clean_signal=[TWOPI,1])
        phobos_mean_rotational_rate = default_phobos_mean_rotational_rate  # In rad/s

        plt.figure()
        plt.scatter(librations[:,2] * 360.0 / TWOPI, librations[:,1] * 360.0 / TWOPI)
        plt.grid()
        plt.title('Sub-martian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs / 86400.0, librations[:,1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs / 86400.0, librations[:,2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Sub-martian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.loglog(librations_fourier[:,0] * 86400.0, librations[:,2] * 360 / TWOPI, marker='.')
        plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobian mean motion')
        plt.title(r'Libration frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        plt.grid()
        plt.legend()

    # Accelerations exerted by all third bodies. This will be used to assess whether the bodies are needed or not.
    if checks[4]:
        third_body_accelerations = dependents[:,[0,16,17,18,19,20]]
        third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
        plt.figure()
        for idx, body in enumerate(third_bodies):
            plt.semilogy(epochs / 86400.0, third_body_accelerations[:, idx + 1], label=body)
        plt.title('Third body accelerations')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel(r'Acceleration [m/s²]')
        plt.legend()
        plt.grid()
        
        
def run_model_a2_checks(checks: list[int],
                        bodies: numerical_simulation.environment.SystemOfBodies,
                        damping_results: numerical_simulation.propagation.RotationalProperModeDampingResults,
                        check_undamped: bool) -> None:
    
    if sum(checks) > 0:

        mars_mu = bodies.get('Mars').gravitational_parameter
        states = dict2array(damping_results.forward_backward_states[-1][1])
        dependents = dict2array(damping_results.forward_backward_dependent_variables[-1][1])

        mean_motion_history = dependents[:,[0,6]]
        mean_motion_history[:,1] = np.sqrt(mars_mu / mean_motion_history[:,1] ** 3)
        average_mean_motion, trash = average_over_integer_number_of_orbits(mean_motion_history,dependents[:,[0,6,7,8,9,10,11]])

        phobos_mean_rotational_rate = default_phobos_mean_rotational_rate

        if check_undamped:
            states_undamped = dict2array(damping_results.forward_backward_states[0][0])
            dependents_undamped = dict2array(damping_results.forward_backward_dependent_variables[0][0])

        epochs_array = states[:,0]

    # Trajectory
    if checks[0]:
        if check_undamped:
            cartesian_history_undamped = array2dict(states_undamped[:,[0,1,2,3,4,5,6]])
            trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
        cartesian_history = array2dict(states[:,[0,1,2,3,4,5,6]])
        trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        if check_undamped:
            keplerian_history_undamped = dependents_undamped[:,[0,7,8,9,10,11,12]]
            plot_kepler_elements(keplerian_history_undamped, title='Undamped COEs')
        keplerian_history = dependents[:,[0,7,8,9,10,11,12]]
        plot_kepler_elements(keplerian_history, title='Damped COEs')

    # Orbit is equatorial
    if checks[2]:
        if check_undamped:
            sub_phobian_point_undamped = dependents_undamped[:,[0,14,15]]
            sub_phobian_point_undamped[:,1:] = bring_inside_bounds(sub_phobian_point_undamped[:,1:], -PI, PI, include='upper')
        sub_phobian_point = dependents[:,[0,14,15]]
        sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:, 1:], -PI, PI, include='upper')

        if check_undamped:
            plt.figure()
            plt.scatter(sub_phobian_point_undamped[:,2] * 360.0 / TWOPI,
                        sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:, 1] * 360.0 / TWOPI,
                        label='Damped', marker='+')
            plt.grid()
            plt.title('Sub-phobian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            plt.title('Undamped sub-phobian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label=r'$Lon$')
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
            euler_history_undamped = dependents_undamped[:,[0,1,2,3]]
            euler_history_undamped[:,1:] = bring_inside_bounds(euler_history_undamped[:,1:], 0.0, TWOPI)
            euler_fourier_undamped = fourier_transform(euler_history_undamped, clean_signal)
        euler_history = dependents[:,[0,1,2,3]]
        euler_history[:,1:] = bring_inside_bounds(euler_history[:,1:], 0.0, TWOPI)
        euler_fourier = fourier_transform(euler_history, clean_signal)

        if check_undamped:
            plt.figure()
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label=r'$\psi$')
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label=r'$\theta$')
            # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
            plt.legend()
            plt.grid()
            # plt.xlim([0.0, 3.5])
            plt.title('Undamped Euler angles')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Angle [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, euler_history[:,1] * 360.0 / TWOPI, label=r'$\psi$')
        plt.plot(epochs_array / 86400.0, euler_history[:,2] * 360.0 / TWOPI, label=r'$\theta$')
        # plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
        plt.legend()
        plt.grid()
        # plt.xlim([0.0, 3.5])
        plt.title('Damped Euler angles')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Angle [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(euler_fourier_undamped[:,0] * 86400.0, euler_fourier_undamped[:,1] * 360.0 / TWOPI, label=r'$\psi$', marker='.')
            plt.loglog(euler_fourier_undamped[:,0] * 86400.0, euler_fourier_undamped[:,1] * 360.0 / TWOPI, label=r'$\phi$', marker='.')
            plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
            plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
            plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
            plt.title(r'Undamped frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            # plt.xlim([0, 70])
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(euler_fourier[:,0] * 86400.0, euler_fourier[:,1] * 360 / TWOPI, label=r'$\psi$', marker='.')
        plt.loglog(euler_fourier[:,0] * 86400.0, euler_fourier[:,2] * 360 / TWOPI * 360 / TWOPI, label=r'$\theta$', marker='.')
        plt.loglog(euler_fourier[:,0] * 86400.0, euler_fourier[:,3] * 360 / TWOPI * 360 / TWOPI, label=r'$\phi$', marker='.')
        plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
        plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.title(r'Damped frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        # plt.xlim([0, 70])
        plt.grid()
        plt.legend()

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[4]:
        normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor,
                                                                       phobos_mean_rotational_rate)
        if check_undamped:
            sub_martian_point_undamped = dict2array(extract_elements_from_history(dependents_undamped, [4, 5]))
            sub_martian_point_undamped[:, 1:] = bring_inside_bounds(sub_martian_point_undamped[:, 1:], -PI, PI,
                                                                    include='upper')
            libration_history_undamped = extract_elements_from_history(dependents_undamped, 5)
            libration_freq_undamped, libration_amp_undamped = fourier_transform(
                libration_history_undamped)
        sub_martian_point = dict2array(extract_elements_from_history(dependents, [4, 5]))
        sub_martian_point[:, 1:] = bring_inside_bounds(sub_martian_point[:,1:], -PI, PI, include='upper')
        libration_history = extract_elements_from_history(dependents, 5)
        libration_freq, libration_amp = fourier_transform(libration_history)

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
    #         third_body_torques_undamped = dict2array(
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
    #     third_body_torques = dict2array(extract_elements_from_history(dependents_damped, list(range(18, 23))))
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
                       damping_results: numerical_simulation.propagation.RotationalProperModeDampingResults,
                       check_undamped: bool) -> None:

    if sum(checks) > 0:

        mars_mu = bodies.get('Mars').gravitational_parameter
        states = dict2array(damping_results.forward_backward_states[-1][1])
        dependents = dict2array(damping_results.forward_backward_dependent_variables[-1][1])

        mean_motion_history = dependents[:,[0,6]]
        mean_motion_history[:,1] = np.sqrt(mars_mu / mean_motion_history[:,1] ** 3)
        average_mean_motion, trash = average_over_integer_number_of_orbits(mean_motion_history,dependents[:,[0,6,7,8,9,10,11]])

        phobos_mean_rotational_rate = default_phobos_mean_rotational_rate

        if check_undamped:
            states_undamped = dict2array(damping_results.forward_backward_states[0][0])
            dependents_undamped = dict2array(damping_results.forward_backward_dependent_variables[0][0])

        epochs_array = states[:,0]

    # Trajectory
    if checks[0]:
        if check_undamped:
            cartesian_history_undamped = array2dict(states_undamped[:,[0,1,2,3,4,5,6]])
            trajectory_3d(cartesian_history_undamped, ['Phobos'], 'Mars')
        cartesian_history = array2dict(states[:,[0,1,2,3,4,5,6]])
        trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        if check_undamped:
            keplerian_history_undamped = dependents_undamped[:,[0,7,8,9,10,11,12]]
            plot_kepler_elements(keplerian_history_undamped, title='Undamped COEs')
        keplerian_history = dependents[:,[0,7,8,9,10,11,12]]
        plot_kepler_elements(keplerian_history, title='Damped COEs')

    # Orbit is equatorial
    if checks[2]:
        if check_undamped:
            sub_phobian_point_undamped = dependents_undamped[:,[0,14,15]]
            sub_phobian_point_undamped[:,1:] = bring_inside_bounds(sub_phobian_point_undamped[:,1:], -PI, PI, include='upper')
        sub_phobian_point = dependents[:,[0,14,15]]
        sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:, 1:], -PI, PI, include='upper')

        if check_undamped:
            plt.figure()
            plt.scatter(sub_phobian_point_undamped[:,2] * 360.0 / TWOPI,
                        sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:, 1] * 360.0 / TWOPI,
                        label='Damped', marker='+')
            plt.grid()
            plt.title('Sub-phobian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, sub_phobian_point_undamped[:,2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            plt.title('Undamped sub-phobian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label=r'$Lon$')
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
            euler_history_undamped = dependents_undamped[:,[0,1,2,3]]
            euler_history_undamped[:,1:] = bring_inside_bounds(euler_history_undamped[:,1:], 0.0, TWOPI)
            euler_fourier_undamped = fourier_transform(euler_history_undamped, clean_signal)
        euler_history = dependents[:,[0,1,2,3]]
        euler_history[:,1:] = bring_inside_bounds(euler_history[:,1:], 0.0, TWOPI)
        euler_fourier = fourier_transform(euler_history, clean_signal)

        if check_undamped:
            plt.figure()
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:,1] * 360.0 / TWOPI, label=r'$\psi$')
            plt.plot(epochs_array / 86400.0, euler_history_undamped[:,2] * 360.0 / TWOPI, label=r'$\theta$')
            # plt.plot(epochs_array / 86400.0, euler_history_undamped[:,3] * 360.0 / TWOPI, label = r'$\phi$')
            plt.legend()
            plt.grid()
            # plt.xlim([0.0, 3.5])
            plt.title('Undamped Euler angles')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Angle [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, euler_history[:,1] * 360.0 / TWOPI, label=r'$\psi$')
        plt.plot(epochs_array / 86400.0, euler_history[:,2] * 360.0 / TWOPI, label=r'$\theta$')
        # plt.plot(epochs_array / 86400.0, euler_history[:,3] * 360.0 / TWOPI, label = r'$\phi$')
        plt.legend()
        plt.grid()
        # plt.xlim([0.0, 3.5])
        plt.title('Damped Euler angles')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Angle [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(euler_fourier_undamped[:,0] * 86400.0, euler_fourier_undamped[:,1] * 360.0 / TWOPI, label=r'$\psi$', marker='.')
            plt.loglog(euler_fourier_undamped[:,0] * 86400.0, euler_fourier_undamped[:,1] * 360.0 / TWOPI, label=r'$\phi$', marker='.')
            plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
            plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
            plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
            plt.title(r'Undamped frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            # plt.xlim([0, 70])
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(euler_fourier[:,0] * 86400.0, euler_fourier[:,1] * 360 / TWOPI, label=r'$\psi$', marker='.')
        plt.loglog(euler_fourier[:,0] * 86400.0, euler_fourier[:,2] * 360 / TWOPI * 360 / TWOPI, label=r'$\theta$', marker='.')
        plt.loglog(euler_fourier[:,0] * 86400.0, euler_fourier[:,3] * 360 / TWOPI * 360 / TWOPI, label=r'$\phi$', marker='.')
        plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
        plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.title(r'Damped frequency content')
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        # plt.xlim([0, 70])
        plt.grid()
        plt.legend()

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[4]:

        clean_signal = [TWOPI, 1]
        if check_undamped:
            librations_undamped = dependents_undamped[:,[0,5,6]]
            librations_undamped[:,1:] = bring_inside_bounds(librations_undamped[:,1:], -PI, PI, include='upper')
            librations_fourier_undamped = fourier_transform(librations_undamped, clean_signal)
        librations = dependents[:,[0,5,6]]
        librations[:,1:] = bring_inside_bounds(librations[:,1:], -PI, PI, include='upper')
        librations_fourier = fourier_transform(librations, clean_signal)

        if check_undamped:
            plt.figure()
            plt.scatter(librations_undamped[:,2] * 360.0 / TWOPI, librations_undamped[:,1] * 360.0 / TWOPI, label='Undamped')
            plt.scatter(librations[:,2] * 360.0 / TWOPI, librations[:,1] * 360.0 / TWOPI, label='Damped', marker='+')
            plt.grid()
            plt.title(r'Sub-martian point')
            plt.xlabel('LON [º]')
            plt.ylabel('LAT [º]')
            plt.legend()

            plt.figure()
            plt.plot(epochs_array / 86400.0, librations_undamped[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
            plt.plot(epochs_array / 86400.0, librations_undamped[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
            plt.legend()
            plt.grid()
            # plt.title('Undamped sub-martian point ($\omega = ' + str(phobos_mean_rotational_rate) + '$ rad/s)')
            plt.title('Undamped sub-martian point')
            plt.xlabel('Time [days since J2000]')
            plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.scatter(librations[:, 2] * 360.0 / TWOPI, librations[:, 1] * 360.0 / TWOPI)
        plt.grid()
        plt.title(r'Damped sub-martian point')
        plt.xlabel('LON [º]')
        plt.ylabel('LAT [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, librations[:, 1] * 360.0 / TWOPI, label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, librations[:, 2] * 360.0 / TWOPI, label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Damped sub-martian point')
        plt.xlabel('Time [days since J2000]')
        plt.ylabel('Coordinate [º]')

        if check_undamped:
            plt.figure()
            plt.loglog(librations_fourier_undamped[:,0] * 86400.0, librations_fourier_undamped[:,2] * 360 / TWOPI, marker='.', label='Lon')
            plt.loglog(librations_fourier_undamped[:,0] * 86400.0, librations_fourier_undamped[:,1] * 360 / TWOPI, marker='.', label='Lat')
            plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
            plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
            plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
            plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
            plt.title(r'Undamped libration frequency content')
            plt.xlabel(r'$\omega$ [rad/day]')
            plt.ylabel(r'$A [º]$')
            plt.grid()
            plt.legend()

        plt.figure()
        plt.loglog(librations_fourier[:,0] * 86400.0, librations_fourier[:,2] * 360 / TWOPI, marker='.', label='Lon')
        plt.loglog(librations_fourier[:,0] * 86400.0, librations_fourier[:,1] * 360 / TWOPI, marker='.', label='Lat')
        plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
        plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
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
    #         third_body_torques_undamped = dict2array(
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
    #     third_body_torques = dict2array(extract_elements_from_history(dependents_damped, list(range(18, 23))))
    #     plt.figure()
    #     for idx, body in enumerate(third_bodies):
    #         plt.semilogy(epochs_array / 86400.0, third_body_torques[:, idx + 1], label=body)
    #     plt.title('Third body torques')
    #     plt.xlabel('Time [days since J2000]')
    #     plt.ylabel(r'Torque [N$\cdot$m]')
    #     plt.yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
    #     plt.legend()
    #     plt.grid()

    plt.show()


def plot_kepler_elements(keplerian_history: np.ndarray, title: str = '') -> None:

    epochs_array = keplerian_history[:,0]

    (fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))) = plt.subplots(3, 2)
    # Semi-major axis
    ax1.plot(epochs_array / 86400.0, keplerian_history[:,1] / 1000.0)
    ax1.set_xlabel('Time [days since J2000]')
    ax1.set_ylabel(r'$a$ [km]')
    ax1.set_title('Semimajor axis')
    ax1.grid()
    # Eccentricity
    ax2.plot(epochs_array / 86400.0, keplerian_history[:,2])
    ax2.set_xlabel('Time [days since J2000]')
    ax2.set_ylabel(r'$e$ [-]')
    ax2.set_title('Eccentricity')
    ax2.grid()
    # Inclination
    ax3.plot(epochs_array / 86400.0, keplerian_history[:,4] * 360.0 / TWOPI)
    ax3.set_xlabel('Time [days since J2000]')
    ax3.set_ylabel(r'$i$ [º]')
    ax3.set_title('Inclination')
    ax3.grid()
    # Right-ascension of ascending node
    ax4.plot(epochs_array / 86400.0, keplerian_history[:,3] * 360.0 / TWOPI)
    ax4.set_xlabel('Time [days since J2000]')
    ax4.set_ylabel(r'$\Omega$ [º]')
    ax4.set_title('RAAN')
    ax4.grid()
    # Argument of periapsis
    ax5.plot(epochs_array / 86400.0, keplerian_history[:,5] * 360.0 / TWOPI)
    ax5.set_xlabel('Time [days since J2000]')
    ax5.set_ylabel(r'$\omega$ [º]')
    ax5.set_title('Argument of periapsis')
    ax5.grid()
    # True anomaly
    ax6.plot(epochs_array / 86400.0, keplerian_history[:,6] * 360.0 / TWOPI)
    ax6.set_xlabel('Time [days since J2000]')
    ax6.set_ylabel(r'$\theta$ [º]')
    ax6.set_title('True anomaly')
    ax6.grid()

    fig.tight_layout()
    if title == '': fig.suptitle('Keplerian elements')
    else: fig.suptitle(title)

    return


'''

########################################################################################################################
########################################################################################################################
##########                                                                                                    ##########
##########                                  NUMERICAL DIFFERENTIATION                                         ##########
##########                                                                                                    ##########
########################################################################################################################
########################################################################################################################

'''


def compute_numerical_partials_in_state_transition_matrix(bodies: numerical_simulation.environment.SystemOfBodies,
                                                          propagator_settings: propagation_setup.propagator.PropagatorSettings,
                                                          perturbation_vector: np.ndarray) -> dict[float, np.ndarray] :

    '''

    !!! UNFINISHED !!!

    :param bodies:
    :param propagator_settings:
    :param perturbation_vector:
    :return:
    '''

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

    '''

    !!! UNFINISHED !!!

    :param bodies:
    :param propagator_settings:
    :param perturbation_vector:
    :return:
    '''

    return


def modify_libration_column_of_design_matrix(analytical_design_matrix: np.ndarray,
                                             observation_times: list,
                                             derivative_history: dict) -> tuple:

    design_matrix = analytical_design_matrix.copy()
    K = max(abs(derivative_history))
    for idx, epoch in enumerate(observation_times):
        design_matrix[3 * idx:3 * (idx + 1), -1] = derivative_history[epoch][:3] / K

    # numerical_observations = np.zeros(len(observation_collection.concatenated_observations))
    # observation_times = [observation_collection.concatenated_times[idx] for idx in range(len(observation_collection.concatenated_times)) if idx % 3 == 0]
    # for idx, epoch in enumerate(observation_times):
    #     numerical_observations[3*idx:3*(idx+1)] = simulator.state_history[epoch][:3]
    # numerical_post_residuals = observation_collection.concatenated_observations - numerical_observations

    return design_matrix, K


def compute_numerical_partials_wrt_scaled_libration_amplitude(bodies: numerical_simulation.environment.SystemOfBodies,
                                                              initial_epoch: float,
                                                              initial_state: np.ndarray,
                                                              simulation_time: float,
                                                              step : float = 0.001) -> dict:

    pre_fit_libration_amplitude = bodies.get( 'Phobos').rotation_model.libration_calculator.get_scaled_libration_amplitude()

    bodies.get('Phobos').rotation_model.libration_calculator = numerical_simulation.environment.DirectLongitudeLibrationCalculator(pre_fit_libration_amplitude-step)
    propagator_settings = get_propagator_settings('A1', bodies, initial_epoch, initial_state, simulation_time)
    simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
    states_left = simulator.state_history

    bodies.get('Phobos').rotation_model.libration_calculator = numerical_simulation.environment.DirectLongitudeLibrationCalculator(pre_fit_libration_amplitude+step)
    propagator_settings = get_propagator_settings('A1', bodies, initial_epoch, initial_state, simulation_time)
    simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
    states_right = simulator.state_history

    derivative_history = dict2array(compare_results(states_right,
                                                      states_left,
                                                      list(states_right.keys())))
    derivative_history[:,1:] = derivative_history[:,1:] / (2*step)

    return array2dict(derivative_history)