import os
import pyvista as pv
import torch

rf_sols = []
for f in os.listdir("./checkpoints/"):
    if str(f)[0] == ".":
        continue
    print(f)
    time = float(str(f).split("checkpoint_T_")[1].split('.vtk')[0])
    f = os.path.join("checkpoints/", f)
    rf_sol = pv.read(f)
    rf_sol['rho_n'] = rf_sol['rho_n']*6 - 5
    rf_sols.append([rf_sol,time])
rf_sols = sorted(rf_sols, key=lambda x: x[1])


dataset = []
for (rf, t) in rf_sols:
    print(t)
    pts = torch.stack([torch.ones(len(rf.points))*t, torch.tensor(rf.points[:,0]), torch.tensor(rf.points[:,1])]).T
    u_ref = torch.tensor(rf['u_n'])[:,:2]
    rho_ref = torch.tensor(rf['rho_n'])
    y_ref = torch.cat([rho_ref.unsqueeze(1), rho_ref.unsqueeze(1)*u_ref], dim=1)
    dataset.append((pts, y_ref))
    
    
# Plot the data for every value of t in the true_figures folder
if not os.path.exists('true_figures'):
    os.makedirs('true_figures')

# Determine the global min and max values for the color scale
rho_min = min(rf['rho_n'].min() for rf, _ in rf_sols)
rho_max = max(rf['rho_n'].max() for rf, _ in rf_sols)

from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset
import numpy as np
from scipy.interpolate import griddata
pts_tensor = torch.cat([item[0] for item in dataset])
y_ref_tensor = torch.cat([item[1] for item in dataset])

np.save('data/pts.npy', pts_tensor.numpy())
np.save('data/y_ref.npy', y_ref_tensor.numpy())

print(pts_tensor.shape)
print(y_ref_tensor.shape)


# Create a meshgrid of numpy arrays for plotting
# Create a meshgrid of numpy arrays for plotting
x = np.linspace(pts_tensor[:, 1].min(), pts_tensor[:, 1].max(), 100)
y = np.linspace(pts_tensor[:, 2].min(), pts_tensor[:, 2].max(), 100)
X, Y = np.meshgrid(x, y)


# Interpolate the data on the meshgrid

dataset_sol = []
for t in sorted(set(pts_tensor[:, 0].numpy())):
    print(t)
    mask = pts_tensor[:, 0] == t
    points = pts_tensor[mask][:, 1:].numpy()
    values = y_ref_tensor[mask][:, 0].numpy()  # Assuming we want to interpolate rho_n

    Z = griddata(points, values, (X, Y), method='linear')
    u_sol = griddata(points, y_ref_tensor[mask][:, 1].numpy(), (X, Y), method='linear')
    v_sol = griddata(points, y_ref_tensor[mask][:, 2].numpy(), (X, Y), method='linear')
    dataset_sol.append(torch.column_stack([torch.ones_like(torch.tensor(X)).ravel()*t, torch.tensor(X).ravel(), torch.tensor(Y).ravel(), torch.tensor(Z).ravel(), torch.tensor(u_sol).ravel(), torch.tensor(v_sol).ravel()]))

    plt.figure()
    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title(f'Time = {t.item()}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f'true_figures/figure_{t.item():.2f}.png')
    plt.close()
# Convert the list of tuples into a single tensor for pts and y_ref
pts_tensor = torch.cat([item[0] for item in dataset])
y_ref_tensor = torch.cat([item[1] for item in dataset])

print(pts_tensor.shape)
print(y_ref_tensor.shape)

# Save the dataset
dataset = TensorDataset(pts_tensor, y_ref_tensor)
if not os.path.exists('data'):
    os.makedirs('data')

torch.save(dataset, 'data/sol_dataset.pt')
print("Dataset saved!")

np.save('data/solution.npy', np.concatenate(dataset_sol))
print(np.concatenate(dataset_sol).shape)
