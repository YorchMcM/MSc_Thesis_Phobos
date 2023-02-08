import numpy as np
# import fuckit
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel import constants

def normalize_spherical_harmonic_coefficients(cosine_coefficients: np.ndarray, sine_coefficients: np.ndarray) -> tuple:

    max_degree, max_order = cosine_coefficients.shape

    for degree in range(int(max_degree + 1)):
        for order in range(int(max_order + 1)):
            if order == 0 : delta = 1
            else : delta = 0
            N = np.sqrt((2 - delta)*(2*order+1)*np.math.factorial(order - degree)/np.math.factorial(order + degree))
            cosine_coefficients[degree, order] = cosine_coefficients[degree, order] / N  # Should this be a times or an over?
            sine_coefficients[degree, order] = sine_coefficients[degree, order] / N  # Should this be a times or an over?

    return cosine_coefficients, sine_coefficients


def let_there_be_a_gravitational_field(frame_name: str) -> environment_setup.gravity_field.GravityFieldSettings:

    # I set the frame_name to be an input because this has to be consistent with other parts of the program, so that it
    # is easy to check that from the main script without coming here, and can be easily changed if necessary.

    phobos_gravitational_parameter = 713000.0
    phobos_reference_radius = 11118.81652
    phobos_normalized_cosine_coefficients = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [-0.04660347700, 0.0, 0.024184276330, 0.0, 0.0],
                                                      [0.002998797015, -0.004139321225, -0.008785040655, 0.001185163133,
                                                       0.0],
                                                      [0.006429537912, 0.003369690127, -0.002323017571, -0.003114272077,
                                                       0.0008212813403]])
    phobos_normalized_sine_coefficients = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.002045708945, 0.001045820499, -0.013200531600, 0.0],
                                                    [0.0, -0.001010497508, -0.001589281757, 0.002661315051,
                                                     0.00007342710506]])

    settings_to_return = environment_setup.gravity_field.spherical_harmonic(
        phobos_gravitational_parameter,
        phobos_reference_radius,
        phobos_normalized_cosine_coefficients,
        phobos_normalized_sine_coefficients,
        associated_reference_frame = frame_name)

    return settings_to_return


def inertia_tensor_from_spherical_harmonic_gravity_field_settings(
        gravity_field_settings: environment_setup.gravity_field.SphericalHarmonicsGravityFieldSettings) -> np.ndarray:

    try:
        C_20 = gravity_field_settings.normalized_cosine_coefficients[2,0]
        C_22 = gravity_field_settings.normalized_cosine_coefficients[2,0]
    except:
        raise ValueError('Insufficient spherical harmonics for the computation of an inertia tensor.')

    R = gravity_field_settings.reference_radius
    M = gravity_field_settings.gravitational_parameter / constants.GRAVITATIONAL_CONSTANT

    N_20 = np.sqrt(5)
    N_22 = np.sqrt(10)

    C_20 = C_20 / N_20  # Should this be a times or an over?
    C_22 = C_22 / N_22  # Should this be a times or an over?

    aux = M*R**2
    I = (2/5)*aux
    A = aux * (C_20/3 - 2*C_22) + I
    B = aux * (C_20/3 + 2*C_22) + I
    C = aux * (-2*C_20/3) + I

    inertia_tensor = np.array([[A, 0, 0], [0, B, 0], [0, 0, C]])

    return inertia_tensor