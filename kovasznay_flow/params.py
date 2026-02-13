import numpy as np

# Equation parameters
L = 1.
Re = 50  # Reynolds Number
nu = 1 / Re  # Viscosity
lam = 1/2*Re - np.sqrt(0.25*Re**2 + 4*np.pi**2)  # Eigenvalue of the problem


# Domain parameters
x_min = -1
x_max = 1.
y_min = -0.5    
y_max = 1.5
dx = 0.01
dy = 0.01
# Define boundary points
nx = int((x_max - x_min) / dx) + 1
ny = int((y_max - y_min) / dy) + 1
# Number of points
N = 10000
