'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
'''

# import os
# from datetime import datetime

from Auxiliaries import *

# from tudatpy.kernel.numerical_simulation import estimation, estimation_setup, Estimator
# from tudatpy.kernel.numerical_simulation.estimation_setup import observation, parameter

simulation_time = 30.0 * constants.JULIAN_DAY

trajectory_file = '/home/yorch/thesis/phobos-ephemerides-3500-nolib.txt'
phobos_ephemerides = get_ephemeris_from_file(trajectory_file)
bodies = get_solar_system(phobos_ephemerides, scaled_amplitude = 0.0)

interpolator = bodies.get('Phobos').ephemeris.interpolator
ephemeris_epochs = interpolator.get_independent_values()
ephemeris_states = interpolator.get_dependent_values()
ephemeris_history = dict(zip(ephemeris_epochs, ephemeris_states))

control_initial_epoch = 0.0
control_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(control_initial_epoch)
control_propagator_settings = get_model_a1_propagator_settings(bodies,
                                                               simulation_time = simulation_time,
                                                               initial_epoch = control_initial_epoch,
                                                               initial_state = control_initial_state)
control_simulator = numerical_simulation.create_dynamics_simulator(bodies, control_propagator_settings)
control_state_history = control_simulator.state_history
epochs_list = list(control_state_history.keys())
control_differences = dict.fromkeys(epochs_list)
for epoch in epochs_list:
    control_differences[epoch] = control_state_history[epoch] - interpolator.interpolate(epoch)
control_differences_array = result2array(control_differences)
control_norms = np.zeros([len(epochs_list), 3])
control_norms[:,0] = np.array(epochs_list)
for idx in range(len(epochs_list)):
    epoch = epochs_list[idx]
    control_norms[idx,1] = np.linalg.norm(control_differences[epoch][:3])
    control_norms[idx,2] = np.linalg.norm(control_differences[epoch][3:])

plt.figure()
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_differences_array[:, 1], label='x', marker='.')
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_differences_array[:, 2], label='x', marker='.')
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_differences_array[:, 3], label='x', marker='.')
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_norms[:, 1], label='Norm', marker='.')
plt.grid()
plt.xlabel('Time since J2000 [days]')
plt.ylabel(r'$\Delta\vec r$ [m]')
plt.legend()
plt.title('Control differences')

plt.figure()
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_differences_array[:, 4], label='x', marker='.')
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_differences_array[:, 5], label='x', marker='.')
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_differences_array[:, 6], label='x', marker='.')
plt.semilogy(control_differences_array[:, 0] / 86400.0, control_norms[:, 2], label='Norm', marker='.')
plt.grid()
plt.xlabel('Time since J2000 [days]')
plt.ylabel(r'$\Delta\vec v$ [m/s]')
plt.legend()
plt.title('Control differences')

test_initial_epoch = 30.0 * constants.JULIAN_DAY
test_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(test_initial_epoch)

test_propagator_settings = get_model_a1_propagator_settings(bodies,
                                                            simulation_time = simulation_time,
                                                            initial_epoch = test_initial_epoch,
                                                            initial_state = test_initial_state)
test_simulator = numerical_simulation.create_dynamics_simulator(bodies, test_propagator_settings)
test_state_history = test_simulator.state_history
epochs_list = list(test_state_history.keys())

differences = dict.fromkeys(epochs_list)

for epoch in list(test_state_history.keys()):
    differences[epoch] = test_state_history[epoch] - interpolator.interpolate(epoch)

differences_array = result2array(differences)
norms = np.zeros([len(epochs_list), 3])
norms[:,0] = np.array(epochs_list)
for idx in range(len(epochs_list)):
    epoch = epochs_list[idx]
    norms[idx,1] = np.linalg.norm(differences[epoch][:3])
    norms[idx,2] = np.linalg.norm(differences[epoch][3:])

plt.figure()
plt.semilogy(differences_array[:,0] / 86400.0, differences_array[:,1], label='x', marker='.')
plt.semilogy(differences_array[:,0] / 86400.0, differences_array[:,2], label='x', marker='.')
plt.semilogy(differences_array[:,0] / 86400.0, differences_array[:,3], label='x', marker='.')
plt.semilogy(differences_array[:,0] / 86400.0, norms[:,1], label='Norm', marker='.')
plt.grid()
plt.xlabel('Time since J2000 [days]')
plt.ylabel(r'$\Delta\vec r$ [m]')
plt.legend()
plt.title('Test differences')

plt.figure()
plt.semilogy(differences_array[:,0] / 86400.0, differences_array[:,4], label='x', marker='.')
plt.semilogy(differences_array[:,0] / 86400.0, differences_array[:,5], label='x', marker='.')
plt.semilogy(differences_array[:,0] / 86400.0, differences_array[:,6], label='x', marker='.')
plt.semilogy(differences_array[:,0] / 86400.0, norms[:,2], label='Norm', marker='.')
plt.grid()
plt.xlabel('Time since J2000 [days]')
plt.ylabel(r'$\Delta\vec v$ [m/s]')
plt.legend()
plt.title('Test differences')


