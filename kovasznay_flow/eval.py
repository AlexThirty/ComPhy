import torch
import numpy as np

from params import x_min, x_max, y_min, y_max, nx, ny
from models.pinn import PINN
from models.ncl import NCL
from models.pinnncl import PINN_Ncl
from models.dualpinn import DualPINN
device = 'cuda:0'
import matplotlib.pyplot as plt

# Set the font size for matplotlib
plt.rcParams.update({'font.size': 14})

# Define the models
pinn_model_static = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
pinn_model_resample = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
pinn_model_grad = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
ncl_model_static = NCL(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
ncl_model_ntk = NCL(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
ncl_model_grad = NCL(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_derl_static = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_derl_ntk = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_derl_grad = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)

dualpinn_outl_static = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_outl_ntk = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_outl_grad = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)

dualpinn_sob_static = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_sob_ntk = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
dualpinn_sob_grad = DualPINN(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)


pinnncl_derl_static = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)
pinnncl_derl_ntk = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)
pinnncl_derl_grad = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)

pinnncl_outl_static = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)
pinnncl_outl_ntk = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)
pinnncl_outl_grad = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)

pinnncl_sob_static = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)
pinnncl_sob_ntk = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)
pinnncl_sob_grad = PINN_Ncl(alignment_mode='none',
    hidden_units=[64 for i in range(4)],
    device=device
).to(device)


# Load the weights
pinn_model_static.load_state_dict(torch.load('saved_models/PINN_none_static.pth'))
pinn_model_resample.load_state_dict(torch.load('saved_models/PINN_resample_static.pth'))
pinn_model_grad.load_state_dict(torch.load('saved_models/PINN_none_grad.pth'))
ncl_model_static.load_state_dict(torch.load('saved_models/NCL_none_static.pth'))
ncl_model_ntk.load_state_dict(torch.load('saved_models/NCL_none_ntk.pth'))
ncl_model_grad.load_state_dict(torch.load('saved_models/NCL_none_grad.pth'))
dualpinn_derl_static.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_static.pth'))
dualpinn_derl_ntk.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_ntk.pth'))
dualpinn_derl_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_grad.pth'))
dualpinn_outl_static.load_state_dict(torch.load('saved_models/MP-2xPINN_OUTL_static.pth'))
dualpinn_outl_ntk.load_state_dict(torch.load('saved_models/MP-2xPINN_OUTL_ntk.pth'))
dualpinn_outl_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_OUTL_grad.pth'))
dualpinn_sob_static.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_static.pth'))
dualpinn_sob_ntk.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_ntk.pth'))
dualpinn_sob_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_grad.pth'))
pinnncl_derl_static.load_state_dict(torch.load('saved_models/MP-PINN+NCL_DERL_static.pth'))
pinnncl_derl_ntk.load_state_dict(torch.load('saved_models/MP-PINN+NCL_DERL_ntk.pth'))
pinnncl_derl_grad.load_state_dict(torch.load('saved_models/MP-PINN+NCL_DERL_grad.pth'))
pinnncl_outl_static.load_state_dict(torch.load('saved_models/MP-PINN+NCL_OUTL_static.pth'))
#pinnncl_outl_ntk.load_state_dict(torch.load('saved_models/MP-PINN+NCL_OUTL_ntk.pth'))
pinnncl_outl_grad.load_state_dict(torch.load('saved_models/MP-PINN+NCL_OUTL_grad.pth'))
pinnncl_sob_static.load_state_dict(torch.load('saved_models/MP-PINN+NCL_SOB_static.pth'))
#pinnncl_sob_ntk.load_state_dict(torch.load('saved_models/MP-PINN+NCL_SOB_ntk.pth'))
pinnncl_sob_grad.load_state_dict(torch.load('saved_models/MP-PINN+NCL_SOB_grad.pth'))

# Move models to the specified device
pinn_model_static.to(device)
pinn_model_resample.to(device)
pinn_model_grad.to(device)
ncl_model_static.to(device)
ncl_model_ntk.to(device)
ncl_model_grad.to(device)
dualpinn_derl_static.to(device)
dualpinn_derl_ntk.to(device)
dualpinn_derl_grad.to(device)
dualpinn_sob_static.to(device)
dualpinn_sob_ntk.to(device)
dualpinn_sob_grad.to(device)
dualpinn_outl_static.to(device)
dualpinn_outl_ntk.to(device)
dualpinn_outl_grad.to(device)
pinnncl_derl_static.to(device)
pinnncl_derl_ntk.to(device)
pinnncl_derl_grad.to(device)
pinnncl_outl_static.to(device)
pinnncl_outl_ntk.to(device)
pinnncl_outl_grad.to(device)
pinnncl_sob_static.to(device)
pinnncl_sob_ntk.to(device)
pinnncl_sob_grad.to(device)

# Set the models to evaluation mode
pinn_model_static.eval()
pinn_model_resample.eval()
pinn_model_grad.eval()
ncl_model_static.eval()
ncl_model_ntk.eval()
ncl_model_grad.eval()
dualpinn_derl_static.eval()
dualpinn_derl_ntk.eval()
dualpinn_derl_grad.eval()
dualpinn_sob_static.eval()
dualpinn_sob_ntk.eval()
dualpinn_sob_grad.eval()
pinnncl_derl_static.eval()
pinnncl_derl_ntk.eval()
pinnncl_derl_grad.eval()
pinnncl_outl_static.eval()
pinnncl_outl_ntk.eval()
pinnncl_outl_grad.eval()
pinnncl_sob_static.eval()
pinnncl_sob_ntk.eval()
pinnncl_sob_grad.eval()

from plotting import plot_models_errors, generate_grid_data, plot_models_consistencies

# Define the models and their names (NTK ones removed)
models = [
    pinn_model_static, pinn_model_resample, pinn_model_grad,
    ncl_model_static, ncl_model_grad,
    dualpinn_derl_static, dualpinn_derl_grad,
    dualpinn_sob_static, dualpinn_sob_grad,
    dualpinn_outl_static, dualpinn_outl_grad,
    pinnncl_derl_static, pinnncl_derl_grad,
    pinnncl_outl_static, pinnncl_outl_grad,
    pinnncl_sob_static, pinnncl_sob_grad
]
model_names = [
    'PINN static', 'PINN+RAR', 'PINN+Grad',
    'NCL static', 'NCL Grad',
    'MP-2xPINN DERL', 'MP-2xPINN DERL Grad',
    'MP-2xPINN SOB', 'MP-2xPINN SOB Grad',
    'MP-2xPINN OUTL', 'MP-2xPINN OUTL Grad',
    'MP-PINN+NCL DERL', 'MP-PINN+NCL DERL Grad',
    'MP-PINN+NCL OUTL', 'MP-PINN+NCL OUTL Grad',
    'MP-PINN+NCL SOB', 'MP-PINN+NCL SOB Grad'
]

nx = 2*nx
ny = 2*ny
pts, out, vortex = generate_grid_data(nx, ny)
import os
if not os.path.exists('plots/'):
    os.makedirs('plots/')
    

dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
dv = dx * dy

print('nx:', nx)
print('ny:', ny)
print('dv:', dv)
'''
plot_models_errors(models, model_names, pts, out, vortex, 'plots/', nx, ny, dv=dv)
plot_models_consistencies(models, model_names, pts, out, vortex, 'plots/', nx, ny, dv=dv)


import os
if not os.path.exists('plots_best/'):
    os.makedirs('plots_best/')
    
best_models = [
    pinn_model_static, pinn_model_resample, pinn_model_grad,
    ncl_model_static,
    dualpinn_derl_grad,
    pinnncl_derl_grad,
]
best_model_names = [
    'PINN', 'PINN+RAR', 'PINN+Grad',
    'NCL',
    'CP-2xPINN (DERL+Grad)',
    'CP-PINN+NCL (DERL+Grad)',
]

plot_models_errors(best_models, best_model_names, pts, out, vortex, 'plots_best/', nx, ny, dv=dv)
plot_models_consistencies(best_models, best_model_names, pts, out, vortex, 'plots_best/', nx, ny, dv=dv)


from plotting import plot_individual_loss_curves, plot_double_loss_curves
with open('results/MP-2xPINN_DERL_static/losses.pkl', 'rb') as f:
    losses_dualpinn = np.load(f, allow_pickle=True)

#plot_individual_loss_curves(losses_dualpinn, 'MP-2xPINN (DERL)', 'plots_best')


with open('results/MP-PINN+NCL_DERL_static/losses.pkl', 'rb') as f:
    losses_pincl = np.load(f, allow_pickle=True)

#plot_individual_loss_curves(losses_pincl, 'MP-PINN+NCL (DERL)', 'plots_best')


    
plot_double_loss_curves([losses_dualpinn, losses_pincl], ['MP-2xPINN (DERL)', 'MP-PINN+NCL (DERL)'], 'plots_best')
print('PLotted')

from utils import plot_grad_hists

with open('results/MP-2xPINN_DERL_static/grad_values.pkl', 'rb') as f:
    grads_dualpinn_derl = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_dualpinn_derl, save_dir='results/MP-2xPINN_DERL_static', step=99000)

with open('results/PINN_none_static/grad_values.pkl', 'rb') as f:
    grads_pinn = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_pinn, save_dir='results/PINN_none_static', step=99000)

with open('results/NCL_none_static/grad_values.pkl', 'rb') as f:
    grads_ncl = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_ncl, save_dir='results/NCL_none_static', step=99000)

with open('results/MP-2xPINN_OUTL_static/grad_values.pkl', 'rb') as f:
    grads_dualpinn_outl = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_dualpinn_outl, save_dir='results/MP-2xPINN_OUTL_static', step=99000)

with open('results/MP-2xPINN_SOB_static/grad_values.pkl', 'rb') as f:
    grads_dualpinn_sob = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_dualpinn_sob, save_dir='results/MP-2xPINN_SOB_static', step=99000)

with open('results/MP-PINN+NCL_DERL_static/grad_values.pkl', 'rb') as f:
    grads_pinnncl_derl = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_pinnncl_derl, save_dir='results/MP-PINN+NCL_DERL_static', step=99000)

with open('results/MP-PINN+NCL_OUTL_static/grad_values.pkl', 'rb') as f:
    grads_pinnncl_outl = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_pinnncl_outl, save_dir='results/MP-PINN+NCL_OUTL_static', step=99000)

with open('results/MP-PINN+NCL_SOB_static/grad_values.pkl', 'rb') as f:
    grads_pinnncl_sob = np.load(f, allow_pickle=True)
plot_grad_hists(grad_values=grads_pinnncl_sob, save_dir='results/MP-PINN+NCL_SOB_static', step=99000)
'''

from plotting import plot_modules_difference
plot_modules_difference(pinnncl_derl_grad, 'CP-PINN+NCL (DERL+Grad)', pts, out, vortex, 'plots_best/', nx, ny, dv )
plot_modules_difference(dualpinn_derl_grad, 'CP-2xPINN (DERL+Grad)', pts, out, vortex, 'plots_best/', nx, ny, dv, )

plot_modules_difference(pinnncl_sob_grad, 'CP-PINN+NCL (SOB+Grad)', pts, out, vortex, 'plots_best/', nx, ny, dv, )
plot_modules_difference(dualpinn_sob_grad, 'CP-2xPINN (SOB+Grad)', pts, out, vortex, 'plots_best/', nx, ny, dv, )