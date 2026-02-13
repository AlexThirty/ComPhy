import torch
import numpy as np

from params import x_min, x_max, y_min, y_max, nx, ny, nt, t_max, t_min
from models.pinn import PINN
from models.ncl import NCL
from models.pinnncl import PINN_Ncl
from models.dualpinn import DualPINN
device = 'cuda:2'
import matplotlib.pyplot as plt

# Set the font size for matplotlib
plt.rcParams.update({'font.size': 18})

# Define the models
pinn_model_static = PINN(
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
pinn_model_resample = PINN(
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
pinn_model_grad = PINN(
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
ncl_model_static = NCL(
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
ncl_model_grad = NCL(
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
dualpinn_derl_static = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
dualpinn_derl_grad = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device).to(device)

dualpinn_outl_static = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
dualpinn_outl_grad = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device).to(device)

dualpinn_sob_static = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device).to(device)
dualpinn_sob_grad = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device).to(device)


pinnncl_derl_static = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device
).to(device)
pinnncl_derl_grad = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device
).to(device)

pinnncl_outl_static = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device
).to(device)
pinnncl_outl_grad = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device
).to(device)

pinnncl_sob_static = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device
).to(device)
pinnncl_sob_grad = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(2)],
    device=device
).to(device)


# Load the weights
pinn_model_static.load_state_dict(torch.load('saved_models/PINN_none_static.pth'))
pinn_model_resample.load_state_dict(torch.load('saved_models/PINN_resample_static.pth'))
pinn_model_grad.load_state_dict(torch.load('saved_models/PINN_none_grad.pth'))
ncl_model_static.load_state_dict(torch.load('saved_models/NCL_none_static.pth'))
#ncl_model_grad.load_state_dict(torch.load('saved_models/NCL_none_grad.pth'))
dualpinn_derl_static.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_static.pth'))
dualpinn_derl_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_grad.pth'))
dualpinn_outl_static.load_state_dict(torch.load('saved_models/MP-2xPINN_OUTL_static.pth'))
dualpinn_outl_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_OUTL_grad.pth'))
dualpinn_sob_static.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_static.pth'))
dualpinn_sob_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_grad.pth'))
pinnncl_derl_static.load_state_dict(torch.load('saved_models/MP-PINN+NCL_DERL_static.pth'))
pinnncl_derl_grad.load_state_dict(torch.load('saved_models/MP-PINN+NCL_DERL_grad.pth'))
pinnncl_outl_static.load_state_dict(torch.load('saved_models/MP-PINN+NCL_OUTL_static.pth'))
pinnncl_outl_grad.load_state_dict(torch.load('saved_models/MP-PINN+NCL_OUTL_grad.pth'))
pinnncl_sob_static.load_state_dict(torch.load('saved_models/MP-PINN+NCL_SOB_static.pth'))
pinnncl_sob_grad.load_state_dict(torch.load('saved_models/MP-PINN+NCL_SOB_grad.pth'))

# Move models to the specified device
pinn_model_static.to(device)
pinn_model_resample.to(device)
pinn_model_grad.to(device)
ncl_model_static.to(device)
ncl_model_grad.to(device)
dualpinn_derl_static.to(device)
dualpinn_derl_grad.to(device)
dualpinn_sob_static.to(device)
dualpinn_sob_grad.to(device)
dualpinn_outl_static.to(device)
dualpinn_outl_grad.to(device)
pinnncl_derl_static.to(device)
pinnncl_derl_grad.to(device)
pinnncl_outl_static.to(device)
pinnncl_outl_grad.to(device)
pinnncl_sob_static.to(device)
pinnncl_sob_grad.to(device)

# Set the models to evaluation mode
pinn_model_static.eval()
pinn_model_resample.eval()
pinn_model_grad.eval()
ncl_model_static.eval()
ncl_model_grad.eval()
dualpinn_derl_static.eval()
dualpinn_derl_grad.eval()
dualpinn_sob_static.eval()
dualpinn_sob_grad.eval()
pinnncl_derl_static.eval()
pinnncl_derl_grad.eval()
pinnncl_outl_static.eval()
pinnncl_outl_grad.eval()
pinnncl_sob_static.eval()
pinnncl_sob_grad.eval()

from plotting import plot_models_errors, generate_grid_data, plot_models_consistencies, plot_module_distances

# Define the models and their names
models = [pinn_model_static, pinn_model_resample, pinn_model_grad,
          ncl_model_static,
          dualpinn_derl_static, dualpinn_derl_grad,
          dualpinn_sob_static, dualpinn_sob_grad,
          dualpinn_outl_static, dualpinn_outl_grad,
          pinnncl_derl_static, pinnncl_derl_grad,
          pinnncl_outl_static, pinnncl_outl_grad,
          pinnncl_sob_static, pinnncl_sob_grad]
model_names = ['PINN static', 'PINN+RAR', 'PINN+Grad',
               'NCL static',
               'CP-2xPINN (DERL)', 'CP-2xPINN (DERL+Grad)',
               'CP-2xPINN (SOB)', 'CP-2xPINN (SOB+Grad)',
               'CP-2xPINN (OUTL)', 'CP-2xPINN (OUTL+Grad)',
               'CP-PINN+NCL (DERL)', 'CP-PINN+NCL (DERL+Grad)',
               'CP-PINN+NCL (OUTL)', 'CP-PINN+NCL (OUTL+Grad)',
               'CP-PINN+NCL (SOB)', 'CP-PINN+NCL (SOB+Grad)']

nx = 2*nx
ny = 2*ny
nt = 2*nt
pts, out, stream, vortex = generate_grid_data(nx, ny, nt)
import os
if not os.path.exists('plots/'):
    os.makedirs('plots/')
    

dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
dt = (t_max - t_min) / nt
dv = dx * dy * dt

print('nx:', nx)
print('ny:', ny)
print('dv:', dv)

#plot_models_errors(models, model_names, pts, out, vortex, [0., 2, 4, 6, 8], 'plots/', nx, ny, nt, dv=dv)
#plot_models_consistencies(models, model_names, pts, out, vortex, [0., 2, 4, 6, 8], 'plots/', nx, ny, nt, dv=dv)

'''
import os
if not os.path.exists('plots_best/'):
    os.makedirs('plots_best/')
    
best_models = [pinn_model_static, pinn_model_resample, pinn_model_grad, ncl_model_static, dualpinn_derl_grad, pinnncl_derl_grad]
best_model_names = ['PINN', 'PINN+RAR', 'PINN+Grad', 'NCL', 'CP-2xPINN (DERL+Grad)', 'CP-PINN+NCL (DERL+Grad)']
plot_models_errors(best_models, best_model_names, pts, out, vortex, [0, 2, 4, 6, 8], 'plots_best/', nx, ny, nt, dv=dv)
#plot_models_consistencies(best_models, best_model_names, pts, out, vortex, 'plots_best/', nx, ny, dv=dv)


from plotting_utils import plot_individual_loss_curves, plot_double_loss_curves
with open('results_dualpinn_DERL/DERL_test_losses.npy', 'rb') as f:
    losses_dualpinn = np.load(f, allow_pickle=True)

plot_individual_loss_curves(losses_dualpinn, 'MP-2xPINN (DERL)', 'results_dualpinn_DERL')


with open('results_pinnncl_DERL/DERL_test_losses.npy', 'rb') as f:
    losses_pincl = np.load(f, allow_pickle=True)
    
plot_individual_loss_curves(losses_pincl, 'MP-PINN+NCL (DERL)', 'results_pinnncl_DERL')


    
plot_double_loss_curves([losses_dualpinn, losses_pincl], ['MP-2xPINN (DERL)', 'MP-PINN+NCL (DERL)'], 'plots_best')

'''

plot_module_distances(pinnncl_derl_grad, 'CP-PINN+NCL (DERL+Grad)', pts, out, 'plots/', nx, ny, nt, dv=dv)
plot_module_distances(dualpinn_derl_grad, 'CP-2xPINN (DERL+Grad)', pts, out, 'plots/', nx, ny, nt, dv=dv)
plot_module_distances(pinnncl_sob_grad, 'CP-PINN+NCL (SOB+Grad)', pts, out, 'plots/', nx, ny, nt, dv=dv)
plot_module_distances(dualpinn_sob_grad, 'CP-2xPINN (SOB+Grad)', pts, out, 'plots/', nx, ny, nt, dv=dv)