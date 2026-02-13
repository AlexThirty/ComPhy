import numpy as np
import torch
from torch.utils.data import TensorDataset
from params import x_min, x_max, y_min, y_max, t_min, t_max, nu, nx, ny, nt, N

def F(t):
    return torch.exp(-2*nu*t)

def u_true(x: torch.Tensor):
    return torch.sin(x[:,1]) * torch.cos(x[:,2]) * F(x[:,0])

def v_true(x: torch.Tensor):
    return -torch.cos(x[:,1]) * torch.sin(x[:,2]) * F(x[:,0])

def p_true(x: torch.Tensor):
    return 0.25*(torch.cos(2*x[:,1]) + torch.cos(2*x[:,2])) * F(2*x[:,0])

def stream_true(x: torch.Tensor):
    return torch.sin(x[:,1]) * torch.sin(x[:,2]) * F(x[:,0])

def vortex_true(x: torch.Tensor):
    return 2*torch.sin(x[:,1]) * torch.sin(x[:,2]) * F(x[:,0])


def generate_grid_data(nx: int, ny: int, nt: int):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = torch.linspace(x_min, x_max+dx, nx)
    y = torch.linspace(y_min, y_max+dy, ny)
    t = torch.linspace(t_min, t_max+dt, nt)
    X, Y, T = torch.meshgrid(x, y, t)
    pts = torch.stack([T.reshape(-1), X.reshape(-1), Y.reshape(-1)], dim=1)
    print(pts.shape)
    u = u_true(pts)
    v = v_true(pts)
    p = p_true(pts)
    out = torch.stack([u, v, p], dim=1)
    stream = stream_true(pts)
    vortex = vortex_true(pts)
    
    pts_np = pts.detach().numpy()
    u_np = u.detach().numpy()
    v_np = v.detach().numpy()
    p_np = p.detach().numpy()
    stream_np = stream.detach().numpy()
    vortex_np = vortex.detach().numpy()
    out_np = out.detach().numpy()
    
    np.save('data/pts.npy', pts_np)
    np.save('data/u.npy', u_np)
    np.save('data/v.npy', v_np)
    np.save('data/p.npy', p_np)
    np.save('data/stream.npy', stream_np)
    np.save('data/vortex.npy', vortex_np)
    np.save('data/out.npy', out_np)
    print("Grid data saved.")
    print("pts.shape:", pts_np.shape)
    print("out.shape:", out_np.shape)
    print("stream.shape:", stream_np.shape)
    print("vortex.shape:", vortex_np.shape)

    return TensorDataset(pts, out, stream, vortex)


import os
if not os.path.exists('data'):
    os.makedirs('data')

grid_dataset = generate_grid_data(nx, ny, nt)
torch.save(grid_dataset, 'data/sol.pt')

def generate_initial_condition_data(nx: int, ny: int):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x = torch.linspace(x_min, x_max+dx, nx)
    y = torch.linspace(y_min, y_max+dy, ny)
    X, Y = torch.meshgrid(x, y)
    t = torch.zeros(X.numel(), 1)
    pts = torch.cat([t, X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    print(pts.shape)
    u = u_true(pts)
    v = v_true(pts)
    p = p_true(pts)
    out = torch.stack([u, v, p], dim=1)
    stream = stream_true(pts)
    vortex = vortex_true(pts)
    
    pts_np = pts.detach().numpy()
    u_np = u.detach().numpy()
    v_np = v.detach().numpy()
    p_np = p.detach().numpy()
    stream_np = stream.detach().numpy()
    vortex_np = vortex.detach().numpy()
    out_np = out.detach().numpy()
    
    np.save('data/initial_pts.npy', pts_np)
    np.save('data/initial_u.npy', u_np)
    np.save('data/initial_v.npy', v_np)
    np.save('data/initial_p.npy', p_np)
    np.save('data/initial_stream.npy', stream_np)
    np.save('data/initial_vortex.npy', vortex_np)
    np.save('data/initial_out.npy', out_np)
    print("Initial condition data saved.")
    print("pts.shape:", pts_np.shape)
    print("out.shape:", out_np.shape)
    print("stream.shape:", stream_np.shape)
    print("vortex.shape:", vortex_np.shape)

    return TensorDataset(pts, out, stream, vortex)

initial_condition_dataset = generate_initial_condition_data(nx, ny)
torch.save(initial_condition_dataset, 'data/initial_condition.pt')

def generate_boundary_condition_data(nx: int, ny: int, nt: int):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = torch.linspace(x_min, x_max+dx, nx)
    y = torch.linspace(y_min, y_max+dy, ny)
    t = torch.linspace(t_min, t_max+dt, nt)
    
    # Boundary points at x_min and x_max
    X_min, Y_min, T_min = torch.meshgrid(torch.tensor([x_min], dtype=x.dtype), y, t)
    X_max, Y_max, T_max = torch.meshgrid(torch.tensor([x_max]), y, t)
    
    # Boundary points at y_min and y_max
    X_ymin, Y_ymin, T_ymin = torch.meshgrid(x, torch.tensor([y_min], dtype=y.dtype), t)
    X_ymax, Y_ymax, T_ymax = torch.meshgrid(x, torch.tensor([y_max]), t)
    
    # Combine all boundary points
    pts_min = torch.stack([T_min.reshape(-1), X_min.reshape(-1), Y_min.reshape(-1)], dim=1)
    pts_max = torch.stack([T_max.reshape(-1), X_max.reshape(-1), Y_max.reshape(-1)], dim=1)
    pts_ymin = torch.stack([T_ymin.reshape(-1), X_ymin.reshape(-1), Y_ymin.reshape(-1)], dim=1)
    pts_ymax = torch.stack([T_ymax.reshape(-1), X_ymax.reshape(-1), Y_ymax.reshape(-1)], dim=1)
    
    pts = torch.cat([pts_min, pts_max, pts_ymin, pts_ymax], dim=0)
    print(pts.shape)
    
    u = u_true(pts)
    v = v_true(pts)
    p = p_true(pts)
    out = torch.stack([u, v, p], dim=1)
    stream = stream_true(pts)
    vortex = vortex_true(pts)
    
    pts_np = pts.detach().numpy()
    u_np = u.detach().numpy()
    v_np = v.detach().numpy()
    p_np = p.detach().numpy()
    stream_np = stream.detach().numpy()
    vortex_np = vortex.detach().numpy()
    out_np = out.detach().numpy()
    
    np.save('data/boundary_pts.npy', pts_np)
    np.save('data/boundary_u.npy', u_np)
    np.save('data/boundary_v.npy', v_np)
    np.save('data/boundary_p.npy', p_np)
    np.save('data/boundary_stream.npy', stream_np)
    np.save('data/boundary_vortex.npy', vortex_np)
    np.save('data/boundary_out.npy', out_np)
    print("Boundary condition data saved.")
    print("pts.shape:", pts_np.shape)
    print("out.shape:", out_np.shape)
    print("stream.shape:", stream_np.shape)
    print("vortex.shape:", vortex_np.shape)

    return TensorDataset(pts, out, stream, vortex)

boundary_condition_dataset = generate_boundary_condition_data(nx, ny, nt)
torch.save(boundary_condition_dataset, 'data/boundary_condition.pt')



import matplotlib.pyplot as plt

def plot_solution(dataset, times, nx, ny):
    u_min, u_max = float('inf'), float('-inf')
    v_min, v_max = float('inf'), float('-inf')
    p_min, p_max = float('inf'), float('-inf')
    vel_min, vel_max = float('inf'), float('-inf')
    
    for t in times:
        X, Y = torch.meshgrid(torch.linspace(x_min, x_max, nx), torch.linspace(y_min, y_max, ny))
        pts = torch.stack([t*torch.ones_like(X), X, Y], dim=0).reshape(3, -1).T
        u = u_true(pts).reshape(X.shape)
        v = v_true(pts).reshape(X.shape)
        p = p_true(pts).reshape(X.shape)
        vel = torch.sqrt(u**2 + v**2)
        
        u_min, u_max = min(u_min, u.min().item()), max(u_max, u.max().item())
        v_min, v_max = min(v_min, v.min().item()), max(v_max, v.max().item())
        p_min, p_max = min(p_min, p.min().item()), max(p_max, p.max().item())
        vel_min, vel_max = min(vel_min, vel.min().item()), max(vel_max, vel.max().item())
    
    for t in times:
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        pts = np.stack([t*np.ones_like(X), X, Y]).reshape(3, -1).T
        pts = torch.tensor(pts, dtype=torch.float32)
        u = u_true(pts).reshape(X.shape).numpy()
        v = v_true(pts).reshape(X.shape).numpy()
        p = p_true(pts).reshape(X.shape).numpy()
        vel = np.sqrt(u**2 + v**2).reshape(X.shape)
        
        plt.figure(figsize=(32, 8))
        plt.subplot(1, 4, 1)
        plt.contourf(X, Y, u, cmap='jet', vmin=u_min, vmax=u_max, levels=50)
        plt.colorbar()
        plt.title(f'u at t={t}')
        
        plt.subplot(1, 4, 2)
        plt.contourf(X, Y, v, cmap='jet', vmin=v_min, vmax=v_max, levels=50)
        plt.colorbar()
        plt.title(f'v at t={t}')
        
        plt.subplot(1, 4, 3)
        plt.contourf(X, Y, p, cmap='jet', vmin=p_min, vmax=p_max, levels=50)
        plt.colorbar()
        plt.title(f'p at t={t}')

        plt.subplot(1, 4, 4)
        plt.streamplot(X, Y, u, v, color='white', linewidth=1, density=2, arrowstyle='->', arrowsize=1)
        plt.tight_layout()
        plt.contourf(X, Y, vel, cmap='jet', vmin=vel_min, vmax=vel_max, levels=50)
        plt.colorbar()
        plt.title(f'Velocity stream at t={t}')
        
        plt.tight_layout()
        plt.savefig(f'true_figures/solution_t_{t:.2f}.png')
        plt.close()

if not os.path.exists('true_figures'):
    os.makedirs('true_figures')
times = torch.linspace(t_min, t_max, 10)
plot_solution(grid_dataset, times, nx, ny)