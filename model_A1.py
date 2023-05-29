'''

In this script we will define model A1. It propagates the translational dynamics ALONE.

Note: the elements marked with an asterisk (*) are partially or fully defined in this script and are regarded as
something close to "user inputs". The others are fully set somewhere in the Auxiliaries module.

ENVIRONMENT
· Global frame origin: Mars' center of mass
· Global frame orientation: Earth's equator of J2000
* Rotational model: synchronous + once-per-orbit longitudinal libration of amplitude 1.1º (Rambaux et al. 2012)
· Mars' gravity field: default from Tudat
* Phobos' gravity field: From Le Maistre (2019)
· Ephemerides and gravitational parameters of all other bodies: defaults from Tudat

ACCELERATIONS
· Mars' harmonic coefficients up to degree and order 12.
· Phobos' quadrupole gravity field (C20 & C22).
· Third-body point-mass forces by the Sun, Earth, Moon, Deimos, Jupiter and Saturn

PROPAGATOR
· Propagator: Cowell
* Initial epoch: J2000 (01/01/2000 at 12:00)
· Initial state: from spice at initial epoch.
* Simulation time: 3500 days

INTEGRATOR
· Integrator: fixed-step RKDP7(8) with a fixed time step of 5 minutes

'''

from Auxiliaries import *

verbose = True
save_results = False
run_is_for_estimation_check = True

if run_is_for_estimation_check:
    phobos_ephemerides = get_ephemeris_from_file('/home/yorch/thesis/phobos-ephemerides-3500.txt')
    simulation_time = 3.0 * constants.JULIAN_YEAR
    initial_epoch = 1.0 * constants.JULIAN_YEAR
    initial_state = read_vector_history_from_file(getcwd() + '/estimation-ab/alpha/parameter-evolution-a1a1-test-far.txt')[0]
else:
    phobos_ephemerides = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
    simulation_time = 3500.0 * constants.JULIAN_DAY
    initial_epoch = 0.0
    initial_state = None


# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
# The ephemeris model is irrelevant because the translational dynamics of Phobos will be propagated. But tudat complains if Phobos doesn't have one.
if verbose: print('Creating universe...')
if run_is_for_estimation_check: phobos_ephemerides = get_ephemeris_from_file('/home/yorch/thesis/phobos-ephemerides-3500.txt')
else: phobos_ephemerides = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, scaled_amplitude)

# DEFINE PROPAGATION
if verbose: print('Setting up propagation...')
simulation_time = 3500.0*constants.JULIAN_DAY
mutual_spherical = propagation_setup.acceleration.mutual_spherical_harmonic_gravity_type
mars_acceleration_dependent_variable = propagation_setup.dependent_variable.single_acceleration_norm(mutual_spherical, 'Phobos', 'Mars')
dependent_variables_to_save = [ propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('Phobos'),  # 0, 1, 2
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
initial_epoch = 0.0
propagator_settings = get_model_a1_propagator_settings(bodies, simulation_time, initial_epoch, initial_state, dependent_variables_to_save)

# SIMULATE DYNAMICS
tic = time()
if verbose: print('Simulating dynamics...')
simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
if save_results:
    save2txt(simulator.state_history, 'phobos-ephemerides-3500.txt')
    save2txt(simulator.dependent_variable_history, 'a1-dependent-variables-3500.txt')
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
orbital period is of 7h 39min 35.20s.
The tweaked rotational motion in model A2 is of 2.28035245e-4 rad/s = 19.702245168 rad/day. The associated rotational
period is of 7h 39min 13.57s.

This propagated solution is accurate to the 8th significant digit. Performing a longer propagation (of 3500 days) and
averaging over an integer number of orbits (10971 orbits) produces an average mean motion of
2.2785636146219538e-4 rad/s = 19.68678963033368 rad/day. The associate orbital period is 7h 39min 35.20s.
'''