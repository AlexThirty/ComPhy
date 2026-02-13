import numpy as np

def embed(x:np.ndarray):
    c = 2*np.pi
    return np.hstack([np.cos(c*x[:,1:2]), np.sin(c*x[:,1:2]), np.cos(c*x[:,2:3]), np.sin(c*x[:,2:3])])

def generate_random_points(
    N:int, 
    x_min:float, 
    x_max:float, 
    y_min:float, 
    y_max:float,
    t_min:float,
    t_max:float
):
    random_points = np.random.rand(N, 3) * np.array([t_max-t_min, x_max-x_min, y_max-y_min]) + np.array([t_min, x_min, y_min])
    return random_points

def generate_initial_points(
    N:int, 
    x_min:float, 
    x_max:float, 
    y_min:float, 
    y_max:float,
    dt:float
):
    initial_points = np.random.rand(N, 2) * np.array([x_max-x_min, y_max-y_min]) + np.array([x_min, y_min])
    t_0 = np.random.rand(N,1) * dt
    initial_points = np.hstack([t_0, initial_points])
    return initial_points

def true_init(x:np.ndarray):
    z = embed(x)
    rho0 = 1 + (z[:,1] + z[:,3])**2 
    u0 = np.exp(z[:,3])*rho0
    v0 = np.exp(z[:,1])/2*rho0
    return np.hstack([rho0.reshape(-1,1), u0.reshape(-1,1), v0.reshape(-1,1)])

'''
import matplotlib.pyplot as plt

def create_grid(x_min, x_max, y_min, y_max, num_points):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    return grid_points

def plot_initial_condition(grid_points, dt):
    t_0 = np.zeros((grid_points.shape[0], 1))
    initial_points = np.hstack([t_0, grid_points])
    initial_conditions = true_init(initial_points)
    
    plt.figure(figsize=(10, 6))
    plt.quiver(grid_points[:, 0], grid_points[:, 1], initial_conditions[:, 1], initial_conditions[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Initial Condition')
    plt.savefig('initial_condition.png')
    
    plt.figure(figsize=(10, 6))
    plt.contourf(grid_points[:, 0].reshape(100, 100), grid_points[:, 1].reshape(100, 100), initial_conditions[:, 0].reshape(100, 100), cmap='viridis', levels=100)
    plt.colorbar(label='Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Initial Density')
    plt.savefig('initial_density.png')

# Example usage


grid_points = create_grid(0, 1, 0, 1, 100)
plot_initial_condition(grid_points, 0.)

'''