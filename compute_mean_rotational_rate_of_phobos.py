'''

This script will compute the mean rotational rate of Phobos. For that, it will propagate model B several times.

CAUTION : THIS SCRIPT WILL TAKE LONG AS FUCK !!

Once this number is obtained, please hard-code it everywhere else.

The scheme is as follows. Model B is propagated with all its damping iterations. These iterations require a "mean
rotational rate of Phobos". To assess how good this number is, the Phobos-fixed longitude of Mars will be inspected. For
a good value of the mean rotational rate, Mars should not drift to neither East nor West. The longitude time history will
be fit to a straight line, and it will be sought for the slope of this straight line to be as close to 0 as possible.

We will do this by minimizing its square using the fixed point method. Calling the mean rotational rate of Phobos "m" and
the square of slope of the line fitting the longitude as "L", we can construct the function L(m), for which an explicit
expression does not exist. We will take a reference value m_o and compute the slope of L(m) at m = m_o through the numerical
central difference scheme, i.e. dL/dm|m_o = ( L(m_o+h) - L(m_o-h) ) / 2h, where h is a very small number. Using this
derivative, a straight line will be computed that will cross the x-axis at some point. This point will be the reference
m_o for a new iteration.

The process will be considered converged when the slope is below a tolerance "e". Selection of this tolerance will be
done on the basis of ???


'''

from Auxiliaries import *

# Things that do NOT depend on the mean rotational rate of Phobos.
dissipation_times = list(np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]) * 3600.0)  # In seconds.

bodies = get_solar_system('B')
initial_epoch = 0.0
simulation_time = 10.0 * dissipation_times[-1]
dependent_variables = get_list_of_dependent_variables('B', bodies)

what_i_have_until_now = np.array([[0.00022800, 0.00022802, 0.000228035245, 0.00022804, 0.00022806],
                                  [5.9276800826553785e-28, 5.9266320741846535e-28, 5.925816279067625e-28, 5.9255588507623225e-28, 0.0],
                                  [4.11393528e-4, 4.113471594e-4, 4.11328847e-4, 4.113199124e-4, 0.0]]).T

def propagate_and_damp_trajectory(mean_rotational_rate: float) ->  \
        numerical_simulation.propagation.RotationalProperModeDampingResults:

    initial_state = get_undamped_initial_state_at_epoch(bodies, 'B', initial_epoch, mean_rotational_rate)
    propagator_settings = get_propagator_settings('B', bodies, initial_epoch, initial_state, simulation_time,
                                                  dependent_variables)
    damping_results = numerical_simulation.propagation.get_damped_proper_mode_initial_rotational_state(
        bodies,
        propagator_settings,
        mean_rotational_rate,
        dissipation_times)

    dependents = dict2array(damping_results.forward_backward_dependent_variables[-1][1])

    longitude_of_mars = dependents[:,[0,6]]
    coeffs = polyfit(longitude_of_mars[:,0], longitude_of_mars[:,1], 1)
    slope_squared = coeffs[1] ** 2

    return damping_results, slope_squared

rotational_rates = np.array([0.000228000000,
                             0.000228020000,
                             0.000228035245,
                             0.000228040000,
                             0.000228040000])

f = np.zeros(len(rotational_rates))

for idx, rotational_rate in enumerate(rotational_rates):

    print('Performing iteration ' + str(idx+1) + '/' + str(len(rotational_rates)) + ':\nRotational rate: ' + str(rotational_rate))
    damping_results, slope_squared = propagate_and_damp_trajectory(rotational_rate)
    f[idx] = slope_squared
    error = np.sqrt(slope_squared) * simulation_time*360 / TWOPI
    print('Slope squared: ' + str(slope_squared) + ', which is something like ' + str(error) + ' degree deviation in 9 years and 4 months.\n')

plt.figure()
plt.plot(rotational_rates, f, marker = '.')
plt.xlabel(r'Rotational rate, $\omega$ [rad/s]')
plt.ylabel(r'$b^2(\omega)$ [rad/s]')
plt.title(r'Slope $b$ of the longitude of the Phobos-fixed position of Mars')


#
# # Things that depend on the mean rotational rate of Phobos. The iterative process starts here.
# error_in_degrees = 1.0  # ???
# maximum_number_of_iterations = 5
# h = 1e-14
# phobos_mean_rotational_rate = 0.000228035245  # This is the initial guess
#
# tolerance = error_in_degrees * TWOPI / 360.0 / simulation_time
# converged = False
# for iteration in range(maximum_number_of_iterations):
#
#     print('\nITERATION:', iteration)
#     print('CURRENT ROTATIONAL RATE:', phobos_mean_rotational_rate)
#
#     damping_results, slope_squared = propagate_and_damp_trajectory(phobos_mean_rotational_rate)
#
#     print('\nSLOPE SQUARED:', slope_squared)
#
#     if slope_squared > tolerance ** 2 :
#         converged = True
#         print('\nALGORITHM CONVERGED')
#         break
#
#     print('Computing numerical derivative...')
#     damping_results, slope_minus = propagate_and_damp_trajectory(phobos_mean_rotational_rate - h)
#     damping_results, slope_plus = propagate_and_damp_trajectory(phobos_mean_rotational_rate + h)
#     derivative = ( slope_plus - slope_minus ) / (2*h)
#
#     phobos_mean_rotational_rate = phobos_mean_rotational_rate - slope_squared / derivative
#
# if not converged:
#     print('\nITERATION LIMIT REACHED. PROCESS NOT CONVERGED.')
#
# print('\nFinal mean rotational rate:', phobos_mean_rotational_rate)

print('\nPROGRAM COMPLETED SUCCESFULLY')