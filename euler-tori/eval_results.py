import torch
import numpy as np

from models.pinn import PINN
from models.ncl import NCL
from models.pinnncl import PINNNCL
from models.triplepinn import TriplePINN
from models.dualpinn import DualPINN
from models.dualncl import DualNCL
device = 'cpu'
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})


import os
# Define the models
pinn_model = PINN(
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)
pinn_model_grad = PINN(
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)
ncl_model = NCL(
    div_hidden_units=[512 for i in range(8)],
    device=device).to(device)
PINNNCL_derl = PINNNCL(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinn_derl = DualPINN(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualncl_derl = DualNCL(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)

PINNNCL_sob = PINNNCL(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinn_sob = DualPINN(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualncl_sob = DualNCL(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    device=device).to(device)

triple_pinn_derl = TriplePINN(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    mom_hidden_units=[256 for i in range(8)],
    device=device).to(device)

triple_pinn_sob = TriplePINN(
    inc_hidden_units=[256 for i in range(8)],
    div_hidden_units=[256 for i in range(8)],
    mom_hidden_units=[256 for i in range(8)],
    device=device).to(device)


# Load the weights

pinn_model.load_state_dict(torch.load('saved_models/pinn.pt'))
pinn_model_grad.load_state_dict(torch.load('saved_models/pinn_grad.pt'))
ncl_model.load_state_dict(torch.load('saved_models/ncl.pt'))
PINNNCL_derl.load_state_dict(torch.load('saved_models/PINNNCL_DERL.pt'))
dualpinn_derl.load_state_dict(torch.load('saved_models/dualpinn_DERL.pt'))
dualncl_derl.load_state_dict(torch.load('saved_models/dualncl_DERL.pt'))
PINNNCL_sob.load_state_dict(torch.load('saved_models/PINNNCL_SOB.pt'))
dualpinn_sob.load_state_dict(torch.load('saved_models/dualpinn_SOB.pt'))
dualncl_sob.load_state_dict(torch.load('saved_models/dualncl_SOB.pt'))

triple_pinn_derl.load_state_dict(torch.load('saved_models/triplepinn_DERL.pt'))
triple_pinn_sob.load_state_dict(torch.load('saved_models/triplepinn_SOB.pt'))

print('loaded weights')
pinn_model.to(device)
pinn_model_grad.to(device)
ncl_model.to(device)
PINNNCL_derl.to(device)
dualpinn_derl.to(device)
dualncl_derl.to(device)
PINNNCL_sob.to(device)
dualpinn_sob.to(device)
dualncl_sob.to(device)
triple_pinn_derl.to(device)
triple_pinn_sob.to(device)

# Set the models to evaluation mode
pinn_model.eval()
ncl_model.eval()
pinn_model_grad.eval()
PINNNCL_derl.eval()
dualpinn_derl.eval()

from plotting_utils import plot_models_errors, plot_models_consistencies

# Define the models and their names
models = [pinn_model, pinn_model_grad, ncl_model, PINNNCL_derl, dualpinn_derl, dualncl_derl,
         PINNNCL_sob, dualpinn_sob, dualncl_sob, triple_pinn_derl, triple_pinn_sob]
# Define the model names
model_names = ['PINN', 'PINN+Grad', 'NCL', 'CP-2xPINN (DERL)', 'CP-2xNCL (DERL)',
               'CP-2xPINN (SOB)', 'CP-2xNCL (SOB)', 'CP-3xPINN (DERL)', 'CP-3xPINN (SOB)']

import os
if not os.path.exists('plots/'):
    os.makedirs('plots/')


plot_models_errors(models, model_names, [0.001, 0.101, 0.201, 0.301], 'plots/')
plot_models_consistencies(models, model_names, [0.001, 0.101, 0.201, 0.301], 'plots/')


if not os.path.exists('plots_best/'):
    os.makedirs('plots_best/')

models_best = [pinn_model, pinn_model_grad, ncl_model, PINNNCL_derl, dualpinn_derl, dualncl_derl, triple_pinn_derl]
model_names_best = ['PINN', 'PINN+Grad', 'NCL', 'CP-PINN+NCL (DERL)', 'CP-2xPINN (DERL)', 'CP-2xNCL (DERL)', 'CP-3xPINN (DERL)']
plot_models_errors(models_best, model_names_best, [0.001, 0.101, 0.201, 0.301], 'plots_best/')
plot_models_consistencies(models_best, model_names_best, [0.001, 0.101, 0.201, 0.301], 'plots_best/')



from plotting_utils import plot_double_loss_curves

with open('results_PINNNCL/DERL_test_losses.npy', 'rb') as f:
    losses_PINNNCL = np.load(f, allow_pickle=True)

with open('results_dualncl/DERL_test_losses.npy', 'rb') as f:
    losses_dualncl = np.load(f, allow_pickle=True)
    
with open('results_dualpinn/DERL_test_losses.npy', 'rb') as f:
    losses_dualpinn = np.load(f, allow_pickle=True)
    
    
plot_double_loss_curves([losses_dualpinn, losses_PINNNCL, losses_dualncl], ['MP-2xPINN (DERL)', 'MP-PINN+NCL (DERL)', 'MP-2xNCL (DERL)'], 'plots_best')