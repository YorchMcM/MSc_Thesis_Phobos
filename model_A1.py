'''
In this script we will define model A1. It includes:

· Rotational model: synchronous + once-per-orbit longitudinal libration of amplitude 1.1º (Rambaux et al. 2012)
· Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: from spice.
· Simulation time: 90 days
· Integrator: fixed-step RKDP7(8) with a time step of 5 minutes
· Accelerations: Mars' harmonic coefficients up to degree and order 12. Phobos' quadrupole gravity field (C20 & C22).
· Propagator: Cartesian states

'''

# IMPORTS
from matplotlib import use
use('TkAgg')
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from Auxiliaries import *
from time import time

sys.path.insert(0, '/home/yorch/tudat-bundle/cmake-build-release/tudatpy')

from tudatpy.kernel.interface import spice
from tudatpy.io import save2txt

# The following lines set the defaults for plot fonts and font sizes.
for font in fman.findSystemFonts(r'/home/yorch/thesis/Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)
# plt.rc('text', usetex = True)
plt.rc('text.latex', preamble = r'\usepackage{amssymb, wasysym}')

verbose = True

# LOAD SPICE KERNELS
if verbose: print('Loading kernels...')
spice.load_standard_kernels()

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
# The ephemeris model is irrelevant because the translational dynamics of Phobos will be propagated. But tudat complains if Phobos doesn't have one.
if verbose: print('Creating universe...')
phobos_ephemerides = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
gravity_field_type = 'QUAD'
gravity_field_source = 'Le Maistre'
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, gravity_field_type, gravity_field_source, scaled_amplitude)

# DEFINE PROPAGATION
if verbose: print('Setting up propagation...')
simulation_time = 8300.0*constants.JULIAN_DAY
mutual_spherical = propagation_setup.acceleration.mutual_spherical_harmonic_gravity_type
mars_acceleration_dependent_variable = propagation_setup.dependent_variable.single_acceleration_norm(mutual_spherical, 'Phobos', 'Mars')
# dependent_variables_to_save = [ propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos'),  # 0, 1, 2
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Mars', 'Phobos'),  # 3, 4, 5
                                propagation_setup.dependent_variable.keplerian_state('Phobos', 'Mars'),  # 6, 7, 8, 9, 10, 11
                                propagation_setup.dependent_variable.central_body_fixed_spherical_position('Phobos', 'Mars'),  # 12, 13, 14
                                # acceleration_norm_from_body_on_phobos('Sun'), # 15
                                # acceleration_norm_from_body_on_phobos('Earth'),  # 16
                                # # acceleration_norm_from_body_on_phobos('Moon'),  # 17
                                # mars_acceleration_dependent_variable,  # 18
                                # acceleration_norm_from_body_on_phobos('Deimos'),  # 19
                                # acceleration_norm_from_body_on_phobos('Jupiter'),  # 20
                                # acceleration_norm_from_body_on_phobos('Saturn')  # 25
                                ]
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, dependent_variables_to_save)

# SIMULATE DYNAMICS
tic = time()
if verbose: print('Simulating dynamics...')
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
save2txt(simulator.state_history, 'phobos-ephemerides-8300.txt')
save2txt(simulator.dependent_variable_history, 'a1-dependent-variables-8300.txt')
tac = time()
print('SIMULATIONS FINISHED. Time taken:', (tac-tic) / 60.0, 'minutes.')

# POST PROCESS (CHECKS)
checks = [0, 0, 0, 0, 0]
mars_mu = bodies.get('Mars').gravitational_parameter
dependents = simulator.dependent_variable_history
epochs_array = np.array(list(simulator.state_history.keys()))
keplerian_history = extract_elements_from_history(simulator.dependent_variable_history, list(range(6,12)))
average_mean_motion, orbits = average_mean_motion_over_integer_number_of_orbits(keplerian_history, mars_mu)
print('Average mean motion over', orbits, 'orbits:', average_mean_motion, 'rad/s =', average_mean_motion*86400.0, 'rad/day')

# Trajectory
if checks[0]:
    trajectory_3d(simulator.state_history, ['Phobos'], 'Mars')

# Orbit does not blow up.
if checks[1]:
    plot_kepler_elements(keplerian_history)

# Orbit is equatorial
if checks[2]:
    sub_phobian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [13, 14]))
    sub_phobian_point[:,1:] = bring_inside_bounds(sub_phobian_point[:,1:], -PI, PI, include = 'upper')

    plt.figure()
    plt.scatter(sub_phobian_point[:,2] * 360.0 / TWOPI, sub_phobian_point[:,1] * 360.0 / TWOPI)
    plt.grid()
    plt.title('Sub-phobian point')
    plt.xlabel('LON [º]')
    plt.ylabel('LAT [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_phobian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_phobian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.title('Sub-phobian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

# Phobos' x axis points towards Mars with a once-per-orbit longitudinal libration with amplitude as specified above.
if checks[3]:
    sub_martian_point = result2array(extract_elements_from_history(simulator.dependent_variable_history, [4, 5]))
    sub_martian_point[:,1:] = bring_inside_bounds(sub_martian_point[:,1:], -PI, PI, include = 'upper')
    libration_history = extract_elements_from_history(simulator.dependent_variable_history, 5)
    libration_freq, libration_amp = get_fourier_elements_from_history(libration_history)
    # phobos_mean_rotational_rate = 0.00022785759213999574  # In rad/s
    phobos_mean_rotational_rate = 0.000227995  # In rad/s

    plt.figure()
    plt.scatter(sub_martian_point[:,2] * 360.0 / TWOPI, sub_martian_point[:,1] * 360.0 / TWOPI)
    plt.grid()
    plt.title('Sub-martian point')
    plt.xlabel('LON [º]')
    plt.ylabel('LAT [º]')

    plt.figure()
    plt.plot(epochs_array / 86400.0, sub_martian_point[:,1] * 360.0 / TWOPI, label = r'$Lat$')
    plt.plot(epochs_array / 86400.0, sub_martian_point[:,2] * 360.0 / TWOPI, label = r'$Lon$')
    plt.legend()
    plt.grid()
    plt.title('Sub-martian point')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel('Coordinate [º]')

    plt.figure()
    plt.loglog(libration_freq * 86400.0, libration_amp * 360 / TWOPI, marker = '.')
    plt.axline((phobos_mean_rotational_rate * 86400.0, 0),(phobos_mean_rotational_rate * 86400.0, 1), ls = 'dashed', c = 'r', label = 'Phobian mean motion')
    plt.title(r'Libration frequency content')
    plt.xlabel(r'$\omega$ [rad/day]')
    plt.ylabel(r'$A [º]$')
    plt.grid()
    # plt.xlim([0, 21])
    plt.legend()

# Accelerations exerted by all third bodies. This will be used to assess whether the bodies are needed or not.
if checks[4]:
    third_body_accelerations = result2array(extract_elements_from_history(dependents, list(range(15,20))))
    third_bodies = ['Sun', 'Earth', 'Moon', 'Mars', 'Deimos', 'Jupiter', 'Saturn']
    plt.figure()
    for idx, body in enumerate(third_bodies):
        plt.semilogy(epochs_array / 86400.0, third_body_accelerations[:,idx+1], label = body)
    plt.title('Third body accelerations')
    plt.xlabel('Time [days since J2000]')
    plt.ylabel(r'Acceleration [m/s²]')
    plt.legend()
    plt.grid()


'''    
This propagation gives an average mean motion of 2.278563609852602e-4 rad/s = 19.68678958912648 rad/day. The associated
orbital period is of 7h 39min 35.20s. (This solution is "accurate" up to the 8th significant digit.)
The tweaked rotational motion in model A2 is of 2.28035245e-4 rad/s = 19.702245168 rad/day. The associated rotational
period is of 7h 39min 13.57s.
'''