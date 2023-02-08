# IMPORTS
import numpy as np
from matplotlib import pyplot as plt
import fuckit
from Auxiliaries import *

from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, estimation_setup
from tudatpy.kernel import constants
from tudatpy.util import result2array

# LOAD SPICE KERNELS
spice.load_standard_kernels()

# CREATE YOUR UNIVERSE (I USE DEFAULTS FROM SPICE)
bodies_to_create = ["Mars"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# BUILT-IN INFORMATION ON PHOBOS IS QUITE CRAP. WE WILL REMAKE THE WHOLE BODY OF PHOBOS OURSELVES BASED ON SCHEERES (2019).
body_settings.add_empty_settings('Phobos')
# We will begin with the gravitational field.
body_settings.get('Phobos').gravity_field_settings = let_there_be_a_gravitational_field('Phobos_body_fixed')

# For model A1, this is all we need. We will create the rotation model for model A1.
# phobos_rotation = environment_setup.rotation_model.synchronous('Mars', 'J2000', 'Phobos_body_fixed')
body_settings.get('Phobos').rotation_model_settings = environment_setup.rotation_model.synchronous('Mars',
                                                                                                   'J2000',
                                                                                                   'Phobos_body_fixed') # THIS BREAKS

# And then the inertia properties.
phobos_inertia_tensor = inertia_tensor_from_spherical_harmonic_gravity_field_settings(
                                                                    body_settings.get('Phobos').gravity_field_settings)
bodies = environment_setup.create_system_of_bodies(body_settings)