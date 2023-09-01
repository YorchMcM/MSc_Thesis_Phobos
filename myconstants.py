from numpy import pi as PI
from numpy import sqrt
from tudatpy.kernel import constants
from Logistics import read_vector_history_from_file, extract_elements_from_history
from Auxiliaries import average_mean_motion_over_integer_number_of_orbits, get_synodic_period, default_phobos_mean_rotational_rate
TWOPI = 2*PI

dependents = read_vector_history_from_file('a1-dependent-variables-3500.txt')
keplerian_history = extract_elements_from_history(dependents, list(range(6,12)))

mars_mu = 42828375815756.1

# PERIODS OF DIFFERENT BODIES
average_mean_motion, orbits = average_mean_motion_over_integer_number_of_orbits(keplerian_history, mars_mu)
period_phobos = TWOPI / average_mean_motion
period_deimos = TWOPI*sqrt(23463200**3 / mars_mu)
period_mars = 686.98 * 86400.0
period_earth = 365.25 * 86400.0
period_jupiter = 11.862615 * constants.JULIAN_YEAR

# SYNODIC PERIODS
phobos_deimos_synodic_period = get_synodic_period(period_phobos, period_deimos)
mars_earth_synodic_period = get_synodic_period(period_mars, period_earth)
mars_jupiter_synodic_period = get_synodic_period(period_mars, period_jupiter)

# FREQUENCIES
mars_frequency = TWOPI / period_mars
phobos_deimos_frequency = TWOPI / phobos_deimos_synodic_period
mars_earth_frequency = TWOPI / mars_earth_synodic_period
mars_jupiter_frequency = TWOPI / mars_jupiter_synodic_period

# OTHERS
phobos_mean_rotational_rate = default_phobos_mean_rotational_rate  # In rad/s (more of this number, longitude slope goes down)
normal_mode = 0.00011542198506841888