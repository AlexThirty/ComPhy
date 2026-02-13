import numpy as np
import matplotlib.pyplot as plt
import torch
from params import L, Re, lam, nu, x_min, x_max, y_min, y_max, dx, dy, nx, ny
# Load the data
    
def u_true(x:torch.Tensor):
    return torch.column_stack((1 - torch.exp(lam * x[:, 0]/L) * torch.cos(2 * np.pi * x[:, 1]/L),
                      lam/(2*np.pi) * torch.exp(lam*x[:, 0]/L)* torch.sin(2 * np.pi * x[:, 1]/L)))

def vorticity_true(x: torch.Tensor):
    return Re*lam*torch.exp(lam*x[:, 0]/L)*torch.sin(2*np.pi*x[:, 1]/L)/(2*np.pi)

def stream_function(x: torch.Tensor):
    return x[:, 1]/L - 1/(2*np.pi)*torch.exp(lam*x[:, 0]/L)*torch.sin(2*np.pi*x[:, 1]/L)


def p_e_expr(x):
    """Expression for the exact pressure solution to Kovasznay flow"""
    return (1 / 2) * (1 - np.exp(2 * lam * x[0]/L))

def p_true(x:torch.Tensor):
    return 0.5 * (1 - torch.exp(2 * lam * x[:, 0]/L))


# Create a grid in the unit square
x = np.arange(x_min, x_max+dx, dx)
y = np.arange(y_min, y_max+dy, dy)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T


# Evaluate u and p on the grid
u_exact = u_true(torch.tensor(points)).detach().numpy()
p_exact = p_true(torch.tensor(points)).detach().numpy()

vorticity_exact = vorticity_true(torch.tensor(points)).detach().numpy()
stream_exact = stream_function(torch.tensor(points)).detach().numpy()

import os
if not os.path.exists('true_figures'):
    os.makedirs('true_figures')

# Plot the exact velocity field using streamplot
# Plot the exact velocity using quiver and contourf for magnitude
velocity_magnitude = np.sqrt(u_exact[:,0]**2 + u_exact[:,1]**2)
plt.figure(figsize=(10, 5))
plt.streamplot(X, Y, u_exact[:, 0].reshape(X.shape), u_exact[:, 1].reshape(X.shape), color='white', linewidth=1)
plt.contourf(X, Y, velocity_magnitude.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='Velocity magnitude')
plt.title('Exact Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/velocity_field.png')

# Plot the exact pressure field
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, p_exact.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='Pressure')
plt.title('Exact Pressure Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/pressure_field.png')

# Plot the vorticity field
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, vorticity_exact.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Vorticity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/vorticity_field.png')

# Plot the streamline field
plt.figure(figsize=(10, 5))
plt.contour(X, Y, stream_exact.reshape(X.shape), levels=100, cmap='viridis')
plt.colorbar(label='Streamline')
plt.title('Streamline Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('true_figures/streamline_field.png')


# Save in a dataset
from torch.utils.data import TensorDataset
import torch

x_sol = torch.tensor(points).float()
u_sol = u_true(x_sol)
p_sol = p_true(x_sol)
y_sol = torch.concat((u_sol, p_sol.unsqueeze(1)), dim=1)
vorticity_sol = vorticity_true(x_sol)
stream_sol = stream_function(x_sol)

import os
if not os.path.exists('data'):
    os.makedirs('data')
    
dataset = TensorDataset(x_sol, y_sol, vorticity_sol, stream_sol)
torch.save(dataset, 'data/sol.pt')
print('Dataset saved successfully!')

# Save numpy arrays
np.save('data/x_sol.npy', x_sol.numpy())
np.save('data/y_sol.npy', y_sol.detach().numpy())
np.save('data/vorticity_sol.npy', vorticity_sol.detach().numpy())
np.save('data/stream_sol.npy', stream_sol.detach().numpy())

print('Numpy arrays saved successfully!')


boundary_points = np.vstack([
    np.column_stack([np.arange(x_min, x_max + dx, dx), np.full(nx, y_min)]),  # Bottom boundary
    np.column_stack([np.arange(x_min, x_max + dx, dx), np.full(nx, y_max)]),  # Top boundary
    np.column_stack([np.full(ny, x_min), np.arange(y_min, y_max + dy, dy)]),  # Left boundary
    np.column_stack([np.full(ny, x_max), np.arange(y_min, y_max + dy, dy)])   # Right boundary
])

# Evaluate u, p, vorticity, and stream function on the boundary points
u_boundary = u_true(torch.tensor(boundary_points))
p_boundary = p_true(torch.tensor(boundary_points))
y_boundary = torch.concat((u_boundary, p_boundary.unsqueeze(1)), dim=1)
vorticity_boundary = vorticity_true(torch.tensor(boundary_points))
stream_boundary = stream_function(torch.tensor(boundary_points))

boundary_dataset = TensorDataset(torch.tensor(boundary_points), y_boundary, vorticity_boundary, stream_boundary)
torch.save(boundary_dataset, 'data/boundary_condition.pt')
print('Boundary dataset saved successfully!')

# Save the boundary conditions
np.save('data/boundary_points.npy', boundary_points)
np.save('data/y_boundary.npy', y_boundary)
np.save('data/vorticity_boundary.npy', vorticity_boundary)
np.save('data/stream_boundary.npy', stream_boundary)

print('Boundary conditions saved successfully!')