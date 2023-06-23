import sys
import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy.polynomial.polynomial import polyfit
from tudatpy.kernel.interface import spice
from tudatpy.kernel import constants, numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.astro.frame_conversion import inertial_to_rsw_rotation_matrix
from tudatpy.kernel.astro.element_conversion import rotation_matrix_to_quaternion_entries

while '/home/yorch/tudat-bundle/cmake-build-release/tudatpy' in sys.path:
    sys.path.remove('/home/yorch/tudat-bundle/cmake-build-release/tudatpy')

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

from tudatpy.plotting import trajectory_3d

from numpy import pi as PI
TWOPI = 2.0*PI


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

    if original < lower_bound:
        reflect = True
    else:
        reflect = False

    if reflect: original = 2.0 * center - original

    dividend = original - lower_bound
    divisor = upper_bound - lower_bound
    remainder = dividend % divisor
    new = lower_bound + remainder

    if reflect: new = 2.0 * center - new

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


def get_gravitational_field(frame_name: str) -> environment_setup.gravity_field.GravityFieldSettings:

    # The gravitational field implemented here is that by Le Maistre et al. (2019).

    phobos_gravitational_parameter = 1.06e16 * constants.GRAVITATIONAL_CONSTANT
    phobos_reference_radius = 14e3

    phobos_normalized_cosine_coefficients = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.003585, 0.002818, 0.0, 0.0, 0.0],
                                                      [-0.029243, 0.000084, 0.015664, 0.0, 0.0],
                                                      [-0.002222, -0.002450, 0.004268, 0.000917, 0.0],
                                                      [0.002693, -0.001469, -0.000920, 0.001263, 0.000032]])
    phobos_normalized_sine_coefficients = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, -0.002856, 0.0, 0.0, 0.0],
                                                    [0.0, 0.000072, -0.000020, 0.0, 0.0],
                                                    [0.0, 0.001399, -0.000537, -0.006642, 0.0],
                                                    [0.0, 0.000402, -0.000555, -0.001218, 0.000088]])

    settings_to_return = environment_setup.gravity_field.spherical_harmonic(
        phobos_gravitational_parameter,
        phobos_reference_radius,
        phobos_normalized_cosine_coefficients,
        phobos_normalized_sine_coefficients,
        associated_reference_frame=frame_name)

    return settings_to_return


def get_initial_rotational_state_at_epoch(epoch: float) -> np.ndarray:

    translational_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'J2000', 'None', epoch)
    synchronous_rotation_matrix = inertial_to_rsw_rotation_matrix(translational_state).T
    synchronous_rotation_matrix[:,:2] = -1.0 * synchronous_rotation_matrix[:,:2]
    phobos_rotation_quaternion = rotation_matrix_to_quaternion_entries(synchronous_rotation_matrix)

    phobos_mean_rotational_rate = 0.000228035245  # In rad/s
    angular_velocity = np.array([0.0, 0.0, phobos_mean_rotational_rate])

    return np.concatenate((phobos_rotation_quaternion, angular_velocity))

def rotation_matrix_x(angle: float) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), np.sin(angle)],
                     [0.0, -np.sin(angle), np.cos(angle)]])

def rotation_matrix_y(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), 0.0, -np.sin(angle)],
                     [0.0, 1.0, 0.0],
                     [np.sin(angle), 0.0, np.cos(angle)]])

def rotation_matrix_z(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle), 0.0],
                     [-np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])

def euler_angles_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:

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


class KeplerianStates:

    def __init__(self, bodies: numerical_simulation.environment.SystemOfBodies):

        self.GM = bodies.get('Mars').gravitational_parameter
        self.phobos = bodies.get('Phobos')

        return

    def compute_mean_motion(self) -> float:

        state = self.phobos.state

        r_norm = np.sqrt(state[:3] @ state[:3])
        v_dot_v = state[3:] @ state[3:]

        temp_out = 2.0 / r_norm - v_dot_v/self.GM
        temp_in = (2.0*self.GM / r_norm) - v_dot_v

        return temp_out * np.sqrt(temp_in)


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


def get_fourier(time_history: np.ndarray, clean_signal: list = [0.0, 0]) -> tuple:

    '''
    The output of this function will be the frequencies in rad/unit of input, and the amplitudes.
    :param result:
    :param clean_signal:
    :return:
    '''

    sample_times = time_history[:,0]
    signal = time_history[:,1]

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


def get_longitudinal_normal_mode_from_inertia_tensor(inertia_tensor: np.ndarray, mean_motion: float) -> float:

    # From Rambaux (2012) "Rotational motion of Phobos".

    A = inertia_tensor[0,0]
    B = inertia_tensor[1,1]
    C = inertia_tensor[2,2]
    gamma = (B - A) / C

    normal_mode = mean_motion * np.sqrt(3*gamma)

    return normal_mode


def run_checks(bodies: numerical_simulation.environment.SystemOfBodies,
               state_history: np.ndarray,
               dependents_history: np.ndarray,
               checks: list[int],
               title_addition: str = '') -> None:

    if title_addition != '': title_addition = ' ' + title_addition

    mu_mars = bodies.get('Mars').gravitational_parameter
    average_mean_motion = np.mean(dependents_history[:,16])
    phobos_mean_rotational_rate = 0.000228035245  # In rad/s

    epochs_array = state_history[:,0]
    time_label = 'Time since J2000 [days]'
    clean_signal = [TWOPI, 1]

    # Trajectory
    if checks[0]:
        cartesian_history = dict.fromkeys(list(epochs_array))
        for idx, epoch in enumerate(epochs_array):
            cartesian_history[epoch] = state_history[idx,1:7]
        trajectory_3d(cartesian_history, ['Phobos'], 'Mars')

    # Orbit does not blow up.
    if checks[1]:
        plot_kepler_elements(dependents_history[:,:7], title = 'Keplerian elements' + title_addition)

    # Orbit is equatorial
    if checks[2]:
        plt.figure()
        plt.scatter(np.degrees(dependents_history[:,15]), np.degrees(dependents_history[:,14]))
        plt.grid()
        plt.title('Sub-phobian point' + title_addition)
        plt.xlabel('Longitude [º]')
        plt.ylabel('Latitude [º]')

        plt.figure()
        plt.plot(epochs_array / 86400.0, np.degrees(dependents_history[:,14]), label=r'$Lat$')
        plt.plot(epochs_array / 86400.0, np.degrees(dependents_history[:,15]), label=r'$Lon$')
        plt.legend()
        plt.grid()
        plt.title('Sub-phobian point' + title_addition)
        plt.xlabel(time_label)
        plt.ylabel('Coordinate [º]')

    # Phobos' Euler angles.
    if checks[3]:
        normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor,
                                                                       phobos_mean_rotational_rate)
        euler = bring_inside_bounds(dependents_history[:,10:13], 0.0, TWOPI)
        psi_freq, psi_amp = get_fourier(np.hstack((np.atleast_2d(dependents_history[:,0]).T, np.atleast_2d(euler[:,0]).T)), clean_signal)
        theta_freq, theta_amp = get_fourier(np.hstack((np.atleast_2d(dependents_history[:,0]).T, np.atleast_2d(euler[:,1]).T)), clean_signal)
        phi_freq, phi_amp = get_fourier(np.hstack((np.atleast_2d(dependents_history[:,0]).T, np.atleast_2d(euler[:,2]).T)), clean_signal)

        plt.figure()
        plt.plot(epochs_array / 86400.0, np.degrees(euler[:,0]), label=r'$\psi$')
        plt.plot(epochs_array / 86400.0, np.degrees(euler[:,1]), label=r'$\theta$')
        # plt.plot(epochs_array / 86400.0, np.degrees(euler_history[:,2]), label = r'$\phi$')
        plt.legend()
        plt.grid()
        # plt.xlim([0.0, 3.5])
        plt.title('Euler angles w.r.t. the Martian equator' + title_addition)
        plt.xlabel(time_label)
        plt.ylabel('Angle [º]')

        plt.figure()
        plt.loglog(psi_freq * 86400.0, np.degrees(psi_amp), label=r'$\psi$', marker='.')
        plt.loglog(theta_freq * 86400.0, np.degrees(theta_amp), label=r'$\theta$', marker='.')
        plt.loglog(phi_freq * 86400.0, np.degrees(phi_amp), label=r'$\phi$', marker='.')
        plt.ylim([1e-5, np.inf])
        plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed', c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
        plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.title(r'Euler angles frequency content' + title_addition)
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        # plt.xlim([0, 70])
        plt.grid()
        plt.legend()

    # Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
    if checks[4]:
        normal_mode = get_longitudinal_normal_mode_from_inertia_tensor(bodies.get('Phobos').inertia_tensor,
                                                                       phobos_mean_rotational_rate)
        librations = bring_inside_bounds(dependents_history[:,8:10], -PI, PI, 'upper')
        lon_lib_freq, lon_lib_amp = get_fourier(np.hstack((np.atleast_2d(dependents_history[:,0]).T, np.atleast_2d(librations[:,0]).T)), clean_signal)
        lat_lib_freq, lat_lib_amp = get_fourier(np.hstack((np.atleast_2d(dependents_history[:,0]).T, np.atleast_2d(librations[:,1]).T)), clean_signal)

        cmap = plt.get_cmap('PRGn')
        fig, axis = plt.subplots()
        axis.scatter(np.degrees(librations[:,1]), np.degrees(librations[:,0]), c = epochs_array, cmap = cmap)
        axis.grid()
        axis.set_xlabel('Longitude [º]')
        axis.set_ylabel('Latitude [º]')
        axis.set_title('Mars\' coordinates in Phobos\' sky' + title_addition)
        fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(epochs_array[0] / 86400.0, epochs_array[-1] / 86400.0)),
                     ax=axis, orientation='vertical', label=time_label)

        # plt.figure()
        # plt.plot(epochs_array / 86400.0, np.degrees(librations[:,0]), label=r'$Lat$')
        # plt.plot(epochs_array / 86400.0, np.degrees(librations[:,1]), label=r'$Lon$')
        # plt.legend()
        # plt.grid()
        # plt.title('Sub-martian point' + title_addition)
        # plt.xlabel(time_label)
        # plt.ylabel('Coordinate [º]')

        plt.figure()
        plt.loglog(lon_lib_freq * 86400.0, np.degrees(lon_lib_amp), marker='.', label='Lon')
        plt.loglog(lat_lib_freq * 86400.0, np.degrees(lat_lib_amp), marker='.', label='Lat')
        plt.ylim([1e-5, np.inf])
        plt.axvline(phobos_mean_rotational_rate * 86400.0, ls='dashed',
                   c='r', label='Phobos\' mean motion (and integer multiples)')
        plt.axvline(normal_mode * 86400.0, ls='dashed', c='k', label='Longitudinal normal mode')
        plt.axvline(2 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.axvline(3 * average_mean_motion * 86400.0, ls='dashed', c='r')
        plt.title('Libration frequency content' + title_addition)
        plt.xlabel(r'$\omega$ [rad/day]')
        plt.ylabel(r'$A [º]$')
        plt.grid()
        plt.legend()

    # Mean motion
    if checks[5]:
        plt.figure()
        plt.plot(epochs_array / 86400.0, (dependents_history[:,16] - np.mean(dependents_history[:,16])) * 86400.0)
        plt.grid()
        plt.title('Mean motion deviations from average' + title_addition)
        plt.xlabel(time_label)
        plt.ylabel(r'$\Delta n$ [rad/day]')


def plot_kepler_elements(keplerian_history: np.ndarray, title: str = None) -> None:

    epochs_array = keplerian_history[:,0]
    time_label = 'Time since J2000 [days]'

    (fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))) = plt.subplots(3, 2)
    # Semi-major axis
    ax1.plot(epochs_array / 86400.0, keplerian_history[:,1] / 1000.0)
    ax1.set_xlabel(time_label)
    ax1.set_ylabel(r'$a$ [km]')
    ax1.set_title('Semimajor axis')
    ax1.grid()
    # Eccentricity
    ax2.plot(epochs_array / 86400.0, keplerian_history[:,2])
    ax2.set_xlabel(time_label)
    ax2.set_ylabel(r'$e$ [-]')
    ax2.set_title('Eccentricity')
    ax2.grid()
    # Inclination
    ax3.plot(epochs_array / 86400.0, np.degrees(keplerian_history[:,3]))
    ax3.set_xlabel(time_label)
    ax3.set_ylabel(r'$i$ [º]')
    ax3.set_title('Inclination')
    ax3.grid()
    # Right-ascension of ascending node
    ax4.plot(epochs_array / 86400.0, np.degrees(keplerian_history[:,5]))
    ax4.set_xlabel(time_label)
    ax4.set_ylabel(r'$\Omega$ [º]')
    ax4.set_title('RAAN')
    ax4.grid()
    # Argument of periapsis
    ax5.plot(epochs_array / 86400.0, np.degrees(keplerian_history[:,4]))
    ax5.set_xlabel(time_label)
    ax5.set_ylabel(r'$\omega$ [º]')
    ax5.set_title('Argument of periapsis')
    ax5.grid()
    # True anomaly
    ax6.plot(epochs_array / 86400.0, np.degrees(keplerian_history[:,6]))
    ax6.set_xlabel(time_label)
    ax6.set_ylabel(r'$\theta$ [º]')
    ax6.set_title('True anomaly')
    ax6.grid()

    fig.tight_layout()
    if title is None: fig.suptitle('Keplerian elements')
    else: fig.suptitle(title)

    return