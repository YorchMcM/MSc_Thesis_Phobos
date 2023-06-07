'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
'''

from Auxiliaries import get_gravitational_field

from tudatpy.kernel.astro.gravitation import inertia_tensor_from_gravity_field
from tudatpy.kernel.numerical_simulation import environment_setup, environment

# WE FIRST CREATE MARS.
bodies_to_create = ["Mars"]
global_frame_origin = "Mars"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

# WE THEN CREATE PHOBOS.
body_settings.add_empty_settings('Phobos')
body_settings.get('Phobos').gravity_field_settings = get_gravitational_field('Phobos_body_fixed')
bodies = environment_setup.create_system_of_bodies(body_settings)
print(bodies.get('Phobos').inertia_tensor)
I = 0.43  # Mean moment of inertia taken from Rambaux 2012 (no other number found anywhere else)
bodies.get('Phobos').inertia_tensor = inertia_tensor_from_gravity_field(bodies.get('Phobos').gravity_field_model, I)
print(bodies.get('Phobos').inertia_tensor)




