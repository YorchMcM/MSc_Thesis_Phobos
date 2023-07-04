'''
Some scale for things:
    · Semimajor axis : 9500 km
    · Orbital period : 5h
    · Velocity of Phobos : 3 km/s
    · Reference radius: 13 km
'''

from copy import copy, deepcopy
from Auxiliaries import *

# save_dir = os.getcwd() + '/initial-guess-analysis/'
color1, color2, color3, color4 = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
bodies = get_solar_system('S', 'ephemeris/translation-b.eph', 'ephemeris/rotation-b.eph')
R = MarsEquatorOfDate(bodies).j2000_to_mars_rotation
full_state_rotation = np.concatenate((np.concatenate((R, np.zeros([3, 3])), 1), np.concatenate((np.zeros([3, 3]), R), 1)), 0)

trajectory = read_vector_history_from_file('ephemeris/translation-b.eph')
epochs_list = list(trajectory.keys())
state_wrt_mars = np.zeros([len(epochs_list), 7])
inc_ecc = np.zeros([len(epochs_list), 3])

for idx in range(len(epochs_list)):
    current_epoch = epochs_list[idx]
    state_wrt_mars[idx,0] = current_epoch
    inc_ecc[idx,0] = current_epoch

    state_wrt_mars[idx,1:] = full_state_rotation @ trajectory[current_epoch]

    r = state_wrt_mars[idx,1:4]
    v = state_wrt_mars[idx,4:]
    h = np.cross(r,v)

    inc = np.arccos(h[-1] / np.linalg.norm(h))
    ecc = np.linalg.norm(((np.cross(v,h))/42.82837e12) - (r / np.linalg.norm(r)))

    inc_ecc[idx,1:] = np.array([inc, ecc])


theta = np.linspace(0.0, TWOPI, 1001)
circular_trajectory = np.zeros([len(theta), 3])
R = np.mean(np.linalg.norm(state_wrt_mars[:,1:4], axis = 1))
for idx in range(len(theta)):
    angle = theta[idx]
    circular_trajectory[idx] = np.array([R*np.cos(angle), R*np.sin(angle), 0.0])

R_mars = 3390e3
figure, axis = plt.subplots()
axis.add_patch(plt.Circle((0, 0), R_mars / 1e3, color=color2))
axis.plot(state_wrt_mars[:,1] / 1e3, state_wrt_mars[:,2] / 1e3, label = 'Real orbit', c = color3)
axis.plot(circular_trajectory[:,0] / 1e3, circular_trajectory[:,1] / 1e3, label = 'Circular orbit', c = 'purple')
axis.set_xlabel(r'$x$ [km]')
axis.set_ylabel(r'$y$ [km]')
plt.grid()
plt.legend()
axis.set_title('Orbit\'s top view')
plt.axis('equal')

# figure, axis = plt.subplots()
# axis.add_patch(plt.Circle((0, 0), R_mars / 1e3, color=color2))
# axis.plot(state_wrt_mars[:,1] / 1e3, state_wrt_mars[:,3] / 1e3, label = 'Real trajectory', c = color3)
# axis.plot(circular_trajectory[:,0] / 1e3, circular_trajectory[:,2] / 1e3, label = 'Equatorial trajectory', c = 'purple')
# axis.set_xlabel(r'$x$ [km]')
# axis.set_ylabel(r'$z$ [km]')
# plt.grid()
# plt.legend()
# axis.set_title('Side view of Phobos\' trajectory')
# plt.axis('equal')

figure, axis = plt.subplots()
axis.add_patch(plt.Circle((0, 0), R_mars / 1e3, color=color2))
axis.plot(state_wrt_mars[:,2] / 1e3, state_wrt_mars[:,3] / 1e3, label = 'Real orbit', c = color3)
axis.plot(circular_trajectory[:,1] / 1e3, circular_trajectory[:,2] / 1e3, label = 'Equatorial orbit', c = 'purple')
axis.set_xlabel(r'$y$ [km]')
axis.set_ylabel(r'$z$ [km]')
plt.grid()
plt.legend()
axis.set_title('Orbit\'s side view')
plt.axis('equal')

plt.figure()
plt.plot(np.array(epochs_list) / constants.JULIAN_DAY, inc_ecc[:,1], label = r'$i$ [rad]')
plt.plot(np.array(epochs_list) / constants.JULIAN_DAY, inc_ecc[:,2], label = r'$e$ [-]')
plt.xlabel('Time since J2000 [days]')
plt.title('Inclination and eccentricity')
plt.grid()
plt.legend()


print('DONE.')