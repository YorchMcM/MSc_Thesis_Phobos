'''
In this script we will make a study and tuning of the integrator. For that, we will use the following test
simulation:

- Phobos translational dynamics.
- Rotational model: synchronous (no libration)
- Initial epoch: J2000 (01/01/2000 at 12:00)
- Initial state: from spice.
- Simulation time: 90 days
- Accelerations: Mars' harmonic coefficients up to degree and order 12. Phobos' quadrupole gravity field (C20 & C22).
- Cartesian state propagator. (This is the one we will use later also, both for uncoupled and coupled models.)

It is intended that this script is not used anymore once the integrator is selected. However, it will be preserved for
future reference.
'''

# IMPORTS
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fman
from time import time
from os import getcwd
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, estimation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array
from tudatpy.io import save2txt

for font in fman.findSystemFonts(r'C:\Users\Yorch\OneDrive - Delft University of Technology\Year 2022-2023\MSc_Thesis_Phobos\Roboto_Slab'):
    fman.fontManager.addfont(font)

plt.rc('font', family = 'Roboto Slab')
plt.rc('axes', titlesize = 18)
plt.rc('axes', labelsize = 16)
plt.rc('legend', fontsize = 14)

# LOAD SPICE KERNELS
spice.load_standard_kernels()

# CREATE YOUR UNIVERSE (OR AT LEAST MARS, FOR WHAT I USE DEFAULTS FROM SPICE)
bodies_to_create = ["Mars"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# BUILT-IN INFORMATION ON PHOBOS IS QUITE CRAP. WE WILL REMAKE THE WHOLE BODY OF PHOBOS OURSELVES BASED ON LE MAISTRE (2019).
body_settings.add_empty_settings('Phobos')
# Ephemeris and rotation models.
body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
body_settings.get('Phobos').ephemeris_settings = environment_setup.ephemeris.direct_spice('Mars', 'J2000')
# Gravity field.
body_settings.get('Phobos').gravity_field_settings = let_there_be_a_gravitational_field('Phobos_body_fixed', 'QUAD', 'Le Maistre')
# And lastly the list of bodies is created.
bodies = environment_setup.create_system_of_bodies(body_settings)

# NOW WE SET UP THE PROPAGATOR.
bodies_to_propagate = ['Phobos']
central_bodies = ['Mars']
# Acceleration settings
acceleration_settings_on_phobos = dict( Mars = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(12, 12, 2, 2)] )
acceleration_settings = { 'Phobos' : acceleration_settings_on_phobos }
acceleration_model = propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate, central_bodies)
# Initial conditions
initial_epoch = 0.0  # This is the J2000 epoch
initial_state = spice.get_body_cartesian_state_at_epoch('Phobos', 'Mars', 'ECLIPJ2000', 'NONE', initial_epoch)


def generate_benchmark_errors(step_sizes: list[float], simulation_time: float) -> None:

    # Termination settings
    termination_condition = propagation_setup.propagator.time_termination(simulation_time)

    benchmark_errors = dict.fromkeys(step_sizes)
    times_lines = []
    for step_size in step_sizes:
        base_integrator_settings, top_integrator_settings = get_benchmark_integrator_settings(step_size*60.0)
        base_propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_model,
            bodies_to_propagate,
            initial_state,
            initial_epoch,
            base_integrator_settings,
            termination_condition
        )
        top_propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_model,
            bodies_to_propagate,
            initial_state,
            initial_epoch,
            top_integrator_settings,
            termination_condition
        )

        # print('Simulating Phobos for 50 years at ' + str(step_size / 2.0) + '-hour intervals.')
        tic = time()
        base_dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, base_propagator_settings)
        # print('Simulating Phobos for 50 years at ' + str(step_size) + '-hour intervals.')
        tac = time()
        top_dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, top_propagator_settings)
        toe = time()
        times_lines.append(str(step_size) + '    ' + str(tac-tic) + '    ' + str(toe-tac) + '\n')
        # print('dt = ' + str(step_size) + 'min. Base simulator took ' + str(tac-tic) + ' seconds. Top simulator took ' + str(toe-tac) + 's.')
        benchmark_errors[step_size] = compare_results(base_dynamics_simulator.state_history,
                                                      top_dynamics_simulator.state_history,
                                                      list(top_dynamics_simulator.state_history.keys()))

    benchmark_directory = getcwd() + '/benchmark_errors'
    simulation_time = simulation_time / constants.JULIAN_DAY
    if simulation_time > 1000.0: simulation_time = simulation_time / constants.JULIAN_YEAR_IN_DAYS

    times_file = benchmark_directory + '/simulation_times_' + str(simulation_time).removesuffix('.0') + '.txt'
    with open(times_file, 'w') as file:
        for line in times_lines: file.write(line)

    for key in list(benchmark_errors.keys()):
        save2txt(benchmark_errors[key],
                 'benchmark_errors_' + str(simulation_time).removesuffix('.0') + '_' + str(key).removesuffix('.0') + '.txt',
                 benchmark_directory)
