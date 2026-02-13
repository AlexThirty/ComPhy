import torch
import numpy as np

from models.params import x_min, x_max, y_min, y_max, t_min, t_max
from models.pinn import PINN
from models.ncl import NCL
from models.triplencl import TripleNCL
from models.triplepinn import TriplePINN
import os
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})
device = 'cuda:3'

# Define the models
pinn_model = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
pinn_model_resample = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
ncl_model = NCL(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplencl_derl = TripleNCL(alignment_mode='DERL',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplepinn_derl = TriplePINN(alignment_mode='DERL',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplencl_sob = TripleNCL(alignment_mode='SOB',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplepinn_sob = TriplePINN(alignment_mode='SOB',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
pinn_model_grad = PINN(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
ncl_model_grad = NCL(
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplencl_derl_grad = TripleNCL(alignment_mode='DERL',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplepinn_derl_grad = TriplePINN(alignment_mode='DERL',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplencl_sob_grad = TripleNCL(alignment_mode='SOB',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplepinn_sob_grad = TriplePINN(alignment_mode='SOB',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)

# Add output models for triplepinn and triplencl
triplepinn_output = TriplePINN(
    alignment_mode='OUTL',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)
triplencl_output = TripleNCL(
    alignment_mode='OUTL',
    hidden_units=[64 for i in range(4)],
    device=device).to(device)

# Load the weights
pinn_model.load_state_dict(torch.load('saved_models/PINN_none_static.pth'))
#pinn_model_resample.load_state_dict(torch.load('saved_models/PINN_resample_static.pth'))
ncl_model.load_state_dict(torch.load('saved_models/NCL_none_static.pth'))
triplencl_derl.load_state_dict(torch.load('saved_models/MP-3xNCL_DERL_static.pth'))
triplepinn_derl.load_state_dict(torch.load('saved_models/MP-3xPINN_DERL_static.pth'))
triplencl_sob.load_state_dict(torch.load('saved_models/MP-3xNCL_SOB_static.pth'))
triplepinn_sob.load_state_dict(torch.load('saved_models/MP-3xPINN_SOB_static.pth'))
pinn_model_grad.load_state_dict(torch.load('saved_models/PINN_none_grad.pth'))
ncl_model_grad.load_state_dict(torch.load('saved_models/NCL_none_grad.pth'))
triplencl_derl_grad.load_state_dict(torch.load('saved_models/MP-3xNCL_DERL_grad.pth'))
triplepinn_derl_grad.load_state_dict(torch.load('saved_models/MP-3xPINN_DERL_grad.pth'))
triplencl_sob_grad.load_state_dict(torch.load('saved_models/MP-3xNCL_SOB_grad.pth'))
triplepinn_sob_grad.load_state_dict(torch.load('saved_models/MP-3xPINN_SOB_grad.pth'))
triplepinn_output.load_state_dict(torch.load('saved_models/MP-3xPINN_OUTL_static.pth'))
triplencl_output.load_state_dict(torch.load('saved_models/MP-3xNCL_OUTL_static.pth'))

# Set the models to evaluation mode
pinn_model.eval()
ncl_model.eval()
pinn_model_grad.eval()
pinn_model_resample.eval()
ncl_model_grad.eval()
triplencl_derl.eval()
triplepinn_derl.eval()
triplencl_sob.eval()
triplepinn_sob.eval()
triplencl_derl_grad.eval()
triplepinn_derl_grad.eval()
triplencl_sob_grad.eval()
triplepinn_sob_grad.eval()
triplepinn_output.eval()
triplencl_output.eval()

from plotting import plot_models_errors, plot_models_consistencies

# Define the models and their names
models = [
    pinn_model, pinn_model_grad, pinn_model_resample, ncl_model, triplencl_derl, triplepinn_derl,
    triplencl_sob, triplepinn_sob, pinn_model_grad, ncl_model_grad,
    triplencl_derl_grad, triplepinn_derl_grad, triplencl_sob_grad, triplepinn_sob_grad,
    triplepinn_output, triplencl_output
]
model_names = [
    'PINN', 'PINN+Grad', 'PINN+RAR', 'NCL', '3xNCL DERL', '3xPINN DERL', '3xNCL SOB', '3xPINN SOB',
    'PINN (+Grad)', 'NCL (+Grad)', '3xNCL DERL (+Grad)', '3xPINN DERL (+Grad)', '3xNCL SOB (+Grad)', '3xPINN SOB (+Grad)',
    '3xPINN OUTL', '3xNCL OUTL'
]

import os
if not os.path.exists('plots/'):
    os.makedirs('plots/')
    

plot_models_errors(models, model_names, [0., 0.08, 0.16, 0.24], 'plots/')
plot_models_consistencies(models, model_names, [0., 0.08, 0.16, 0.24], 'plots/')

if not os.path.exists('plots_best/'):
    os.makedirs('plots_best/')

models_best = [pinn_model, pinn_model_grad, ncl_model, triplepinn_derl_grad, triplencl_derl_grad]
model_names_best = ['PINN', 'PINN+Grad', 'NCL', 'CP-3xPINN (DERL+Grad)', 'CP-3xNCL (DERL+Grad)']

#plot_models_errors(models_best, model_names_best, [0., 0.08, 0.16, 0.24], 'plots_best/')
#plot_models_consistencies(models_best, model_names_best, [0., 0.08, 0.16, 0.24], 'plots_best/')


from plotting import plot_individual_loss_curves, plot_double_loss_curves

#with open('results/MP-3xPINN_DERL_static/losses.pkl', 'rb') as f:
#    losses_triplepinn = np.load(f, allow_pickle=True)

#plot_individual_loss_curves(losses_triplepinn, 'MP-3xPINN (DERL)', 'results/MP-3xPINN_DERL_static/')


#with open('results/MP-3xNCL_DERL_static/losses.pkl', 'rb') as f:
#    losses_triplencl = np.load(f, allow_pickle=True)

#plot_individual_loss_curves(losses_triplencl, 'MP-3xNCL (DERL)', 'results/MP-3xNCL_DERL_static/')

#plot_double_loss_curves([losses_triplepinn, losses_triplencl], ['MP-3xPINN (DERL)', 'MP-3xNCL (DERL)'], 'plots_best')

#with open('results/MP-3xPINN_SOB_static/losses.pkl', 'rb') as f:
#    losses_triplepinn_sob = np.load(f, allow_pickle=True)

#plot_individual_loss_curves(losses_triplepinn_sob, 'MP-3xPINN (SOB)', 'results/MP-3xPINN_SOB_static/')

#with open('results/MP-3xNCL_SOB_static/losses.pkl', 'rb') as f:
#    losses_triplencl_sob = np.load(f, allow_pickle=True)

#plot_individual_loss_curves(losses_triplencl_sob, 'MP-3xNCL (SOB)', 'results/MP-3xNCL_SOB_static/')
#plot_double_loss_curves([losses_triplepinn_sob, losses_triplencl_sob], ['MP-3xPINN (SOB)', 'MP-3xNCL (SOB)'], 'plots_best')
#plot_double_loss_curves([losses_triplepinn, losses_triplencl], ['MP-3xPINN (DERL)', 'MP-3xNCL (DERL)'], 'plots_best')


def plot_hessian_singular_values(model_names, result_dirs, window=50, save_dir='results/'):
    for model_name, result_dir in zip(model_names, result_dirs):
        hessian_path = os.path.join(result_dir, 'hessian_singular_values.pkl')
        if not os.path.exists(hessian_path):
            print(f"File not found: {hessian_path}")
            continue
        with open(hessian_path, 'rb') as f:
            hess_singular_values = np.load(f, allow_pickle=True)
        print(f"{model_name}: {list(hess_singular_values.keys())}")

        fig, axs = plt.subplots(len(hess_singular_values), 1, figsize=(10, 4 * len(hess_singular_values)))
        if len(hess_singular_values) == 1:
            axs = [axs]
        for idx, (key, values) in enumerate(hess_singular_values.items()):
            values = np.array(values)
            if len(values) < window:
                smoothed = values
            else:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            axs[idx].plot(smoothed)
            axs[idx].set_title(f'{key}')
            axs[idx].set_xlabel('Step')
            axs[idx].set_ylabel('Singular Value')
        plt.tight_layout()
        save_path = os.path.join(result_dir, 'hessian_singular_values_smoothed.png')
        plt.savefig(save_path)
        plt.close(fig)

# Example usage:
model_names = [
    'MP-3xPINN (DERL)', 'PINN', 'NCL', 'MP-3xNCL (DERL)', 'MP-3xNCL (SOB)', 'MP-3xPINN (SOB)'
]
result_dirs = [
    'results/MP-3xPINN_DERL_static',
    'results/PINN_none_static',
    'results/NCL_none_static',
    'results/MP-3xNCL_DERL_static',
    'results/MP-3xNCL_SOB_static',
    'results/MP-3xPINN_SOB_static'
]
#plot_hessian_singular_values(model_names, result_dirs)

from utils import plot_grad_hists

with open('results/MP-3xPINN_DERL_static/grad_values.pkl', 'rb') as f:
    grads_dualpinn_derl = np.load(f, allow_pickle=True)
#with open('results/PINN_none_static/grad_values.pkl', 'rb') as f:
#    grads_pinn = np.load(f, allow_pickle=True)
#with open('results/NCL_none_static/grad_values.pkl', 'rb') as f:
#    grads_ncl = np.load(f, allow_pickle=True)
#with open('results/MP-3xNCL_DERL_static/grad_values.pkl', 'rb') as f:
#    grads_triplencl_derl = np.load(f, allow_pickle=True)
#with open('results/MP-3xNCL_SOB_static/grad_values.pkl', 'rb') as f:
#    grads_triplencl_sob = np.load(f, allow_pickle=True)
#with open('results/MP-3xPINN_SOB_static/grad_values.pkl', 'rb') as f:
#    grads_triplepinn_sob = np.load(f, allow_pickle=True)

plot_grad_hists(grad_values=grads_dualpinn_derl, save_dir='results/MP-3xPINN_DERL_static', step=99000)
#plot_grad_hists(grad_values=grads_pinn, save_dir='results/PINN_none_static', step=99000)
#plot_grad_hists(grad_values=grads_ncl, save_dir='results/NCL_none_static', step=99000)
#plot_grad_hists(grad_values=grads_triplencl_derl, save_dir='results/MP-3xNCL_DERL_static', step=99000)
#plot_grad_hists(grad_values=grads_triplencl_sob, save_dir='results/MP-3xNCL_SOB_static', step=99000)
#plot_grad_hists(grad_values=grads_triplepinn_sob, save_dir='results/MP-3xPINN_SOB_static', step=99000)
