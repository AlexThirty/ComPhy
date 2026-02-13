import numpy as np

rho = 1.0
bulk = 1.0
sound_speed = np.sqrt(bulk/rho)
impedance = rho*sound_speed

x_min = -1.
x_max = 1.
y_min = -1.
y_max = 1.
t_min = 0.
t_max = 0.24
dx = 0.05
dt = 0.01