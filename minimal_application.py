'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
    · Reference radius: 13 km
'''

from Auxiliaries import *

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

save_dir = os.getcwd() + '/initial-guess-analysis/'
color1, color2, color3, color4 = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']

# wrong_states = read_vector_history_from_file(os.getcwd() + '/ephemeris/rotation-b.eph')
# right_states = read_vector_history_from_file(os.getcwd() + '/ephemeris/rotation-b-correct.eph')
#
# wrong_trans = read_vector_history_from_file(os.getcwd() + '/ephemeris/translation-b.eph')
# right_trans = read_vector_history_from_file(os.getcwd() + '/ephemeris/translation-b-correct.eph')
#
# differences = result2array(compare_results(right_states, wrong_states, list(right_states.keys())))
# differences_trans = result2array(compare_results(right_trans, wrong_trans, list(right_trans.keys())))

# plt.figure()
# plt.semilogy(differences[:,0] / 86400.0, abs(differences[:,1]), marker = '.', label = r'$q_0$')
# plt.semilogy(differences[:,0] / 86400.0, abs(differences[:,2]), marker = '.', label = r'$q_1$')
# plt.semilogy(differences[:,0] / 86400.0, abs(differences[:,3]), marker = '.', label = r'$q_2$')
# plt.semilogy(differences[:,0] / 86400.0, abs(differences[:,4]), marker = '.', label = r'$q_3$')
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\Delta q_i$')
# plt.grid()
# plt.legend()
# plt.title('Converged states differences for two different initial guesses of state')
#
# plt.figure()
# plt.semilogy(differences_trans[:,0] / 86400.0, abs(differences_trans[:,1]), marker = '.', label = r'$x$')
# plt.semilogy(differences_trans[:,0] / 86400.0, abs(differences_trans[:,2]), marker = '.', label = r'$y$')
# plt.semilogy(differences_trans[:,0] / 86400.0, abs(differences_trans[:,3]), marker = '.', label = r'$z$')
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\Delta r_i$ [m]')
# plt.grid()
# plt.legend()
# plt.title('Converged states differences for two different initial guesses of state')

simulate_dynamics = False
eph_dir = os.getcwd() + '/ephemeris/'

bodies = get_solar_system('B')

if simulate_dynamics:

    # CORRECT SIMULATION
    print('Simulating correct dynamics...')
    initial_state = np.array([-1.99050285e+06, -8.74300018e+06, -3.18120924e+06,
                               1.84316407e+03, -4.33146556e+01, -1.01843620e+03,
                               7.12119319e-01,  3.07470470e-01,  3.50521934e-02,  6.30174047e-01,
                              -8.14583950e-10,  3.64986777e-09,  2.30331508e-04])
    simulation_time = 294912000.0
    dependent_variables = get_list_of_dependent_variables('B', bodies)
    propagator = get_propagator_settings('B', bodies, 0.0, initial_state, simulation_time, dependent_variables)
    simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator)
    correct_states = simulator.state_history
    correct_dependent_variables = simulator.dependent_variable_history
    save2txt(correct_states, save_dir + 'correct-states.dat')
    save2txt(correct_dependent_variables, save_dir + 'correct-dependents.dat')

    # INCORRECT SIMULATION
    print('Simulating incorrect dynamics...')
    initial_state = np.array([-1.99051385e+06, -8.74299280e+06, -3.18121357e+06,
                              1.84316410e+03, -4.33168477e+01, -1.01843624e+03,
                              7.12119746e-01,  3.07470677e-01,  3.50518130e-02, 6.30173484e-01,
                              -8.14586847e-10,  3.64960444e-09,  2.30331751e-04])
    simulation_time = 294912000.0
    dependent_variables = get_list_of_dependent_variables('B', bodies)
    propagator = get_propagator_settings('B', bodies, 0.0, initial_state, simulation_time, dependent_variables)
    simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator)
    wrong_states = simulator.state_history
    wrong_dependent_variables = simulator.dependent_variable_history
    save2txt(wrong_states, save_dir + 'incorrect-states.dat')
    save2txt(wrong_dependent_variables, save_dir + 'incorrect-dependents.dat')

else:
    correct_states = read_vector_history_from_file(save_dir + 'correct-states.dat')
    correct_dependent_variables = read_vector_history_from_file(save_dir + 'correct-dependents.dat')
    wrong_states = read_vector_history_from_file(save_dir + 'incorrect-states.dat')
    wrong_dependent_variables = read_vector_history_from_file(save_dir + 'incorrect-dependents.dat')



# DIFFERENCES

epochs = np.array(list(correct_states.keys())) / 86400.0
correct_states_array = result2array(correct_states)
incorrect_states_array = result2array(wrong_states)
state_differences = incorrect_states_array - correct_states_array
correct_dependent_variables_array = result2array(correct_dependent_variables)
incorrect_dependent_variables_array = result2array(wrong_dependent_variables)
dependent_variables_differences = incorrect_dependent_variables_array - correct_dependent_variables_array

# # POST PROCESSING
# # Euler angles wrt the Martian equator
# plt.figure()
# plt.plot(epochs, np.degrees(correct_dependent_variables_array[:,1]), marker = '.', label = r'$\Psi$')
# plt.plot(epochs, np.degrees(correct_dependent_variables_array[:,2]), marker = '.', label = r'$\theta$')
# # plt.plot(epochs, np.degrees(correct_dependent_variables_array[:,3]), marker = '.', label = r'$\varphi$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha_i$ [º]')
# plt.title('Euler angles w.r.t. Mars\' equator')

# plt.figure()
# plt.plot(epochs, 3600.0*np.degrees(bring_inside_bounds(dependent_variables_differences[:,1], -PI, PI, 'upper')), marker = '.', label = r'$\Psi$')
# plt.plot(epochs, 3600.0*np.degrees(bring_inside_bounds(dependent_variables_differences[:,2], -PI, PI, 'upper')), marker = '.', label = r'$\theta$')
# # plt.plot(epochs, np.degrees(bring_inside_bounds(dependent_variables_differences[:,3], -PI, PI, 'upper')), marker = '.', label = r'$\varphi$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\Delta\alpha_i$ [arcsec]')
# plt.title('Differences in Euler angles w.r.t. Mars\' equator')

mars_equator = MarsEquatorOfDate(bodies)
for state in correct_dependent_variables_array:
    orbit_euler_angles = np.array([state[11], state[9], state[10]])
    rotated_euler_angles = mars_equator.rotate_euler_angles_from_J2000_to_mars_equator(orbit_euler_angles)
    state[9:12] = np.array([rotated_euler_angles[1], rotated_euler_angles[2], rotated_euler_angles[0]])
for state in incorrect_dependent_variables_array:
    orbit_euler_angles = np.array([state[11], state[9], state[10]])
    rotated_euler_angles = mars_equator.rotate_euler_angles_from_J2000_to_mars_equator(orbit_euler_angles)
    state[9:12] = np.array([rotated_euler_angles[1], rotated_euler_angles[2], rotated_euler_angles[0]])

# # Kepler elements (7, 8, 9, 10, 11, 12)
# plt.figure()
# plt.semilogy(epochs, correct_dependent_variables_array[:,7], marker = '.', label = 'Correct')
# plt.semilogy(epochs, dependent_variables_differences[:,7], marker = '.', label = 'Difference')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$a$ [m]')
# plt.title('Semi major axis')

# plt.figure()
# plt.semilogy(epochs, correct_dependent_variables_array[:,8], marker = '.', label = 'Correct')
# plt.semilogy(epochs, dependent_variables_differences[:,8], marker = '.', label = 'Difference')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$e$ [-]')
# plt.title('Eccentricity')

# plt.figure()
# plt.plot(epochs, np.degrees(correct_dependent_variables_array[:,9]), marker = '.', label = r'$i$')
# plt.plot(epochs, np.degrees(correct_dependent_variables_array[:,10]), marker = '.', label = r'$\omega$')
# plt.plot(epochs, np.degrees(correct_dependent_variables_array[:,11]), marker = '.', label = r'$\Omega$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\alpha_i$ [º]')
# plt.title('Orbit\'s Euler angles w.r.t. Mars\' equator')

# plt.figure()
# plt.plot(epochs, 3600.0*np.degrees(bring_inside_bounds(dependent_variables_differences[:,9], -PI, PI, 'upper')), marker = '.', label = r'$i$')
# # plt.plot(epochs, np.degrees(bring_inside_bounds(dependent_variables_differences[:,10], -PI, PI, 'upper')), marker = '.', label = r'$\omega$')
# plt.plot(epochs, 3600.0*np.degrees(bring_inside_bounds(dependent_variables_differences[:,11], -PI, PI, 'upper')), marker = '.', label = r'$\Omega$')
# plt.grid()
# plt.legend()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\Delta\alpha_i$ [arcsec]')
# plt.title('Differences in orbit\'s Euler angles w.r.t. Mars\' equator')

# plt.figure()
# plt.plot(epochs, np.degrees(bring_inside_bounds(dependent_variables_differences[:,12], -PI, PI, 'upper')), marker = '.')
# plt.grid()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$\Delta\theta$ [º]')
# plt.title('Difference in true anomaly')

mars_mu = 42828375815756.1
correct_mean_motion = np.zeros(len(epochs))
mean_motion_difference = np.zeros(len(epochs))
for idx, state in enumerate(correct_dependent_variables_array):
    correct_mean_motion[idx] = semi_major_axis_to_mean_motion(state[7], mars_mu)
for idx, state in enumerate(incorrect_dependent_variables_array):
    mean_motion_difference[idx] = semi_major_axis_to_mean_motion(state[7], mars_mu)
mean_motion_difference = abs(mean_motion_difference-correct_mean_motion)

plt.figure()
plt.semilogy(epochs, np.degrees(correct_mean_motion) * 86400.0, marker = '.', label = r'$n$')
plt.semilogy(epochs, np.degrees(mean_motion_difference) * 86400.0, marker = '.', label = r'$\Delta n$')
plt.grid()
plt.legend()
plt.xlabel('Time since J2000 [days]')
plt.ylabel(r'$n\ |\ \Delta n$ [º/day]')
plt.title('Mean motion difference and mean motion difference')

# plt.figure()
# plt.semilogy(epochs, norm_rows(state_differences[:,1:4]), marker = '.')
# plt.grid()
# plt.xlabel('Time since J2000 [days]')
# plt.ylabel(r'$|\Delta\vec r|$ [m]')
# plt.title('Position difference')