import torch
import numpy as np

from models.pinn import PINN
from models.ncl import NCL
from models.dualpinn import DualPINN
from models.triplepinn import TriplePINN
from models.dualpinnncl import DualPINN_NCL
from models.quadpinn import QuadPINN
import os
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})
device = 'cuda:2'

# Define the models
pinn_model = PINN(
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
ncl_model = NCL(
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinn_derl = DualPINN(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
triplepinn_derl = TriplePINN(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
quadpinn_derl = QuadPINN(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinn_sob = DualPINN(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
triplepinn_sob = TriplePINN(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
quadpinn_sob = QuadPINN(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
pinn_model_grad = PINN(
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
ncl_model_grad = NCL(
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinn_derl_grad = DualPINN(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
triplepinn_derl_grad = TriplePINN(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
quadpinn_derl_grad = QuadPINN(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinn_sob_grad = DualPINN(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
triplepinn_sob_grad = TriplePINN(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinnncl_derl_static = DualPINN_NCL(alignment_mode='DERL',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
dualpinnncl_sob_static = DualPINN_NCL(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)
quadpinn_sob_grad = QuadPINN(alignment_mode='SOB',
    hidden_units=[256 for i in range(8)],
    device=device).to(device)

# Load the weights
pinn_model.load_state_dict(torch.load('saved_models/PINN_none_static.pth'))
ncl_model.load_state_dict(torch.load('saved_models/NCL_none_static.pth'))
dualpinn_derl.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_static.pth'))
triplepinn_derl.load_state_dict(torch.load('saved_models/MP-3xPINN_DERL_static.pth'))
dualpinn_sob.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_static.pth'))
triplepinn_sob.load_state_dict(torch.load('saved_models/MP-3xPINN_SOB_static.pth'))
dualpinnncl_derl_static.load_state_dict(torch.load('saved_models/MP-2xPINN+NCL_DERL_static.pth'))
dualpinnncl_sob_static.load_state_dict(torch.load('saved_models/MP-2xPINN+NCL_SOB_static.pth'))
quadpinn_derl.load_state_dict(torch.load('saved_models/MP-4xPINN_DERL_static.pth'))
quadpinn_sob.load_state_dict(torch.load('saved_models/MP-4xPINN_SOB_static.pth'))
pinn_model_grad.load_state_dict(torch.load('saved_models/PINN_none_grad.pth'))
ncl_model_grad.load_state_dict(torch.load('saved_models/NCL_none_grad.pth'))
#dualpinn_derl_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_DERL_grad.pth'))
#dualpinn_sob_grad.load_state_dict(torch.load('saved_models/MP-2xPINN_SOB_grad.pth'))
#triplencl_derl_grad.load_state_dict(torch.load('saved_models/MP-3xNCL_DERL_grad.pth'))
#triplepinn_derl_grad.load_state_dict(torch.load('saved_models/MP-3xPINN_DERL_grad.pth'))
#triplencl_sob_grad.load_state_dict(torch.load('saved_models/MP-3xNCL_SOB_grad.pth'))
#triplepinn_sob_grad.load_state_dict(torch.load('saved_models/MP-3xPINN_SOB_grad.pth'))
#quadpinn_derl_grad.load_state_dict(torch.load('saved_models/MP-4xPINN_DERL_grad.pth'))
#quadpinn_sob_grad.load_state_dict(torch.load('saved_models/MP-4xPINN_SOB_grad.pth'))

# Set the models to evaluation mode
pinn_model.eval()
ncl_model.eval()
pinn_model_grad.eval()
ncl_model_grad.eval()
dualpinn_derl.eval()
triplepinn_derl.eval()
dualpinn_sob.eval()
triplepinn_sob.eval()
dualpinn_derl_grad.eval()
triplepinn_derl_grad.eval()
dualpinn_sob_grad.eval()
dualpinnncl_derl_static.eval()
triplepinn_sob_grad.eval()
quadpinn_derl_grad.eval()
quadpinn_sob_grad.eval()

from plotting import plot_models_errors, plot_models_consistencies

# Define the models and their names
models = [pinn_model, pinn_model_grad, ncl_model, dualpinn_derl, triplepinn_derl, dualpinnncl_derl_static, quadpinn_derl, dualpinn_sob, triplepinn_sob, dualpinnncl_sob_static, quadpinn_sob]
          #dualpinn_derl_grad, triplepinn_derl_grad, quadpinn_derl_grad, dualpinn_sob_grad, triplepinn_sob_grad, quadpinn_sob_grad]
model_names = ['PINN', 'PINN (+Grad)', 'NCL', 'CP-2xPINN (DERL)', 'CP-3xPINN (DERL)', 'CP-2xPINN+NCL (DERL)', 'CP-4xPINN (DERL)', 'CP-2xPINN (SOB)', 'CP-3xPINN (SOB)', 'CP-2xPINN+NCL (SOB)', 'CP-4xPINN (SOB)']
               #'CP-2xPINN (DERL+Grad)', 'CP-3xPINN (DERL+Grad)', 'CP-4xPINN (DERL+Grad)', 'CP-2xPINN (SOB+Grad)', 'CP-3xPINN (SOB+Grad)', 'CP-4xPINN (SOB+Grad)']

import os
if not os.path.exists('plots/'):
    os.makedirs('plots/')

t_list = [0.0,0.1,0.2,0.3,0.4,0.5]

plot_models_errors(models, model_names, t_list, 'plots/')
plot_models_consistencies(models, model_names, t_list, 'plots/')

if not os.path.exists('plots_best/'):
    os.makedirs('plots_best/')

models_best = [pinn_model, pinn_model_grad, ncl_model, dualpinn_derl, triplepinn_derl, dualpinnncl_derl_static, quadpinn_derl]
model_names_best = ['PINN', 'PINN+Grad', 'NCL', 'CP-2xPINN (DERL)', 'CP-3xPINN (DERL)', 'CP-2xPINN+NCL (DERL)', 'CP-4xPINN (DERL)']

plot_models_errors(models_best, model_names_best, t_list, 'plots_best/')
plot_models_consistencies(models_best, model_names_best, t_list, 'plots_best/')