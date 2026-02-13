import numpy as np

rho = 1.0
bulk = 1.0
sound_speed = np.sqrt(bulk/rho)
impedance = rho*sound_speed

x_min = 0.
x_max = 1.
y_min = 0.
y_max = 1.
t_min = 0.
t_max = 0.5
nx = 128
ny = 128
nt = 461