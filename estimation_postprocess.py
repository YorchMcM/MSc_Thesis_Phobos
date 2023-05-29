from Auxiliaries import *

read_dir = getcwd() + '/estimation-ab/alpha/'

# CREATE YOUR UNIVERSE. MARS IS ALWAYS THE SAME, WHILE SOME ASPECTS OF PHOBOS ARE TO BE DEFINED IN A PER-MODEL BASIS.
# trajectory_file = '/home/yorch/thesis/everything-works-results/model-b/states-d8192.txt'
phobos_ephemerides = get_ephemeris_from_file('/home/yorch/thesis/phobos-ephemerides-3500.txt')
libration_amplitude = 1.1  # In degrees
ecc_scale = 0.015034167790105173
scaled_amplitude = np.radians(libration_amplitude) / ecc_scale
bodies = get_solar_system(phobos_ephemerides, scaled_amplitude)

initial_estimation_epoch = 1.0 * constants.JULIAN_YEAR
true_initial_state = bodies.get('Phobos').ephemeris.interpolator.interpolate(initial_estimation_epoch)
residual_history = read_vector_history_from_file(read_dir + 'residual-history-a1a1-test-far.txt')
parameter_evolution = read_vector_history_from_file(read_dir + 'parameter-evolution-a1a1-test-far.txt')
residual_rms_evolution = read_vector_history_from_file(read_dir + 'rms-evolution-a1a1-test-far.txt')

number_of_iterations = len(list(parameter_evolution.keys()))

residual_history_array = result2array(residual_history)
parameter_evolution_array = result2array(parameter_evolution)
rms_array = result2array(residual_rms_evolution)

number_of_iterations = int(parameter_evolution_array.shape[0] - 1)
for k in range(number_of_iterations):
    plt.figure()
    plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0, residual_history_array[:,3*k+1] / 1000.0, label = 'x')
    plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0, residual_history_array[:,3*k+2] / 1000.0, label = 'y')
    plt.plot((residual_history_array[:,0] - initial_estimation_epoch) / 86400.0, residual_history_array[:,3*k+3] / 1000.0, label = 'z')
    plt.grid()
    plt.xlabel('Time since estimation start [days]')
    plt.ylabel('Position residuals [km]')
    plt.legend()
    plt.title('Residual history (iteration ' + str(k+1) + ')')

plt.figure()
plt.plot(rms_array[:,0], rms_array[:,1] / 1000.0, label = 'x', marker = '.')
plt.plot(rms_array[:,0], rms_array[:,2] / 1000.0, label = 'y', marker = '.')
plt.plot(rms_array[:,0], rms_array[:,3] / 1000.0, label = 'z', marker = '.')
plt.grid()
plt.xlabel('Iteration number')
plt.ylabel('Residual rms [km]')
plt.legend()
plt.title('Residual root mean square')

plt.figure()
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,1] / 1000.0, label = r'$x_o$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,2] / 1000.0, label = r'$y_o$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,3] / 1000.0, label = r'$z_o$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,4], label = r'$v_{x,o}$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,5], label = r'$v_{y,o}$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,6], label = r'$v_{z,o}$', marker = '.')
plt.grid()
plt.xlabel('Iteration number')
plt.ylabel('Parameter value [km | m/s]')
plt.legend()
plt.title('Parameter history')

plt.figure()
plt.plot(parameter_evolution_array[:,0], (parameter_evolution_array[:,1] - true_initial_state[0]) / 1000.0, label = r'$x_o$', marker = '.')
plt.plot(parameter_evolution_array[:,0], (parameter_evolution_array[:,2] - true_initial_state[1]) / 1000.0, label = r'$y_o$', marker = '.')
plt.plot(parameter_evolution_array[:,0], (parameter_evolution_array[:,3] - true_initial_state[2]) / 1000.0, label = r'$z_o$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,4] - true_initial_state[3], label = r'$v_{x,o}$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,5] - true_initial_state[4], label = r'$v_{y,o}$', marker = '.')
plt.plot(parameter_evolution_array[:,0], parameter_evolution_array[:,6] - true_initial_state[5], label = r'$v_{z,o}$', marker = '.')
plt.grid()
plt.xlabel('Iteration number')
plt.ylabel('Parameter difference from truth [km | m/s]')
plt.legend()
plt.title('Parameter history')

parameter_changes = np.zeros([number_of_iterations, 6])
for k in range(number_of_iterations):
    parameter_changes[k,:] = parameter_evolution[k+1] - parameter_evolution[k]

plt.figure()
plt.plot(parameter_evolution_array[1:,0], abs(parameter_changes[:,0]) / 1000.0, label = r'$x_o$', marker = '.')
plt.plot(parameter_evolution_array[1:,0], abs(parameter_changes[:,1]) / 1000.0, label = r'$y_o$', marker = '.')
plt.plot(parameter_evolution_array[1:,0], abs(parameter_changes[:,2]) / 1000.0, label = r'$z_o$', marker = '.')
plt.plot(parameter_evolution_array[1:,0], abs(parameter_changes[:,3]), label = r'$v_{x,o}$', marker = '.')
plt.plot(parameter_evolution_array[1:,0], abs(parameter_changes[:,4]), label = r'$v_{y,o}$', marker = '.')
plt.plot(parameter_evolution_array[1:,0], abs(parameter_changes[:,5]), label = r'$v_{z,o}$', marker = '.')
plt.yscale('log')
plt.grid()
plt.xlabel('Iteration number')
plt.ylabel('Parameter change [km | m/s]')
plt.legend()
plt.title('Parameter change between pre- and post-fit')

print('PROGRAM COMPLETED SUCCESSFULLY')