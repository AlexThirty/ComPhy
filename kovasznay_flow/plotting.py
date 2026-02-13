from generate import u_true, p_true, vorticity_true
import torch
from params import x_min, x_max, y_min, y_max
import numpy as np
from torch.func import vmap, jacrev, jacfwd, hessian

def generate_grid_data(nx: int, ny: int, return_mesh=False):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    pts = np.column_stack([X.reshape((-1,1)), Y.reshape((-1,1))])
    u = u_true(torch.from_numpy(pts))
    p = p_true(torch.from_numpy(pts)).reshape((-1,1))
    out = torch.column_stack([u, p])
    out = torch.column_stack([u, p])
    vortex = vorticity_true(torch.from_numpy(pts))
    
    vortex_np = vortex.detach().numpy()
    out_np = out.detach().numpy()
    if return_mesh:
        return pts, out_np, vortex_np, X, Y
    
    return pts, out_np, vortex_np


def plot_models_errors(models, model_names, pts, out, vortex, plot_path, nx, ny, dv=1e-3):
    
    pts, out, vortex, X, Y = generate_grid_data(nx, ny, return_mesh=True)
    
    errors_models = {}
    values_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error
        model_out = model.forward(torch.from_numpy(pts).float().to(model.device), return_final=True).cpu().detach().numpy()
        out_err = np.abs(out - model_out)
        # Calculate vorticity error
        model_Dy = vmap(jacrev(model.forward))(torch.from_numpy(pts).float().to(model.device)).cpu().detach().numpy()
        model_vortex = model_Dy[:, 1, 0] - model_Dy[:, 0, 1]
        vortex_err = np.abs(vortex - model_vortex)
        # Save in a dictionary
        errors = {
            'out_err': out_err,
            'vortex_err': vortex_err
        }
        values = {
            'out': model_out,
            'vortex': model_vortex
        }
        
        errors_models[name] = errors
        values_models[name] = values
    
    with open(f'{plot_path}/model_errors.txt', 'w') as f:
        for name, errors in errors_models.items():
            out_err_norm = np.linalg.norm(errors['out_err'], ord=2)*dv
            vortex_err_norm = np.linalg.norm(errors['vortex_err'], ord=2)*dv
            out_err_max = np.max(errors['out_err'])
            vortex_err_max = np.max(errors['vortex_err'])
            f.write(f"{name}:\n")
            f.write(f"Output Error Norm: {out_err_norm}\n")
            f.write(f"Vortex Error Norm: {vortex_err_norm}\n")
            f.write(f"Maximum Output Error: {out_err_max}\n")
            f.write(f"Maximum Vortex Error: {vortex_err_max}\n\n")
    
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, len(models), figsize=(20, 10), layout='compressed', sharex=True, sharey=True)
    
    # Determine the min and max values for the color scales
    out_err_min = np.inf
    out_err_max = -np.inf
    vortex_err_min = np.inf
    vortex_err_max = -np.inf
    
    for name in model_names:
        out_err = errors_models[name]['out_err']
        vortex_err = errors_models[name]['vortex_err']
        out_err_reshaped = np.linalg.norm(out_err.reshape(nx, ny, 3), axis=2)
        vortex_err_reshaped = vortex_err.reshape(nx, ny)
        
        out_err_min = min(out_err_min, out_err_reshaped.min())
        out_err_max = max(out_err_max, out_err_reshaped.max())
        vortex_err_min = min(vortex_err_min, vortex_err_reshaped.min())
        vortex_err_max = max(vortex_err_max, vortex_err_reshaped.max())
    
    norm_out_err = plt.Normalize(vmin=out_err_min, vmax=out_err_max)
    norm_vortex_err = plt.Normalize(vmin=vortex_err_min, vmax=vortex_err_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        out_err = errors_models[name]['out_err']
        vortex_err = errors_models[name]['vortex_err']
        out_err_reshaped = np.linalg.norm(out_err.reshape(nx, ny, 3), axis=2)
        vortex_err_reshaped = vortex_err.reshape(nx, ny)
        
        # Plot output error
        out_err_plot = axs[0, i].contourf(X, Y, out_err_reshaped, cmap='jet', levels=50, norm=norm_out_err, vmin=out_err_min, vmax=out_err_max)
        axs[0, i].set_title(f'{name} Output Error')
        
        # Plot vortex error
        vortex_err_plot = axs[1, i].contourf(X, Y, vortex_err_reshaped, cmap='jet', levels=50, norm=norm_vortex_err, vmin=vortex_err_min, vmax=vortex_err_max)
        axs[1, i].set_title(f'{name} Vortex Error')
    
    # Add a single colorbar for all subplots
    from matplotlib import cm
    #fig.tight_layout()
    fig.colorbar(cm.ScalarMappable(norm=norm_out_err, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.02)
    fig.colorbar(cm.ScalarMappable(norm=norm_vortex_err, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.02)
    
    plt.savefig(f'{plot_path}/errors.png')
    plt.close()
    
    # Plot overall error norm for each model in a multi-column layout
    fig, axs = plt.subplots(1, len(models), figsize=(25, 5), layout='compressed', sharex=True, sharey=True)
    overall_err_min = np.inf
    overall_err_max = -np.inf
    
    # Determine the min and max values for the color scales
    for name in model_names:
        overall_err = np.linalg.norm(errors_models[name]['out_err'].reshape(nx, ny, 3), axis=2)
        overall_err_min = min(overall_err_min, np.nanmin(overall_err))
        overall_err_max = max(overall_err_max, np.nanmax(overall_err))
    
    norm_overall_err = plt.Normalize(vmin=overall_err_min, vmax=overall_err_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        overall_err = np.linalg.norm(errors_models[name]['out_err'].reshape(nx, ny, 3), axis=2)
        overall_err_plot = axs[i].contourf(X, Y, overall_err, cmap='jet', levels=50, norm=norm_overall_err, vmin=overall_err_min, vmax=overall_err_max)
        axs[i].set_title(f'{name}')
    
    # Add a single colorbar for all subplots
    fig.colorbar(cm.ScalarMappable(norm=norm_overall_err, cmap='jet'), ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
    
    
    plt.savefig(f'{plot_path}/overall_error_norm.png')
    plt.close()
    
    
    fig, axs = plt.subplots(2, len(models) + 1, figsize=(20, 10))
    
    # Determine the min and max values for the color scales for model outputs and vortices
    model_out_min = np.inf
    model_out_max = -np.inf
    model_vortex_min = np.inf
    model_vortex_max = -np.inf
    
    for name in model_names:
        model_out = values_models[name]['out']
        model_vortex = values_models[name]['vortex']
        model_out_reshaped = np.linalg.norm(model_out.reshape(nx, ny, 3), axis=2)
        model_vortex_reshaped = model_vortex.reshape(nx, ny)
        
        model_out_min = min(model_out_min, model_out_reshaped.min())
        model_out_max = max(model_out_max, model_out_reshaped.max())
        model_vortex_min = min(model_vortex_min, model_vortex_reshaped.min())
        model_vortex_max = max(model_vortex_max, model_vortex_reshaped.max())
    
    norm_model_out = plt.Normalize(vmin=model_out_min, vmax=model_out_max)
    norm_model_vortex = plt.Normalize(vmin=model_vortex_min, vmax=model_vortex_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        model_out = values_models[name]['out']
        model_vortex = values_models[name]['vortex']
        model_out_reshaped = np.linalg.norm(model_out.reshape(nx, ny, 3), axis=2)
        model_vortex_reshaped = model_vortex.reshape(nx, ny)
        
        # Plot output values
        axs[0, i].contourf(X, Y, model_out_reshaped, cmap='jet', levels=50, norm=norm_model_out, vmin=model_out_min, vmax=model_out_max)
        axs[0, i].set_title(f'{name} OUTL')
        
        # Plot vortex values
        axs[1, i].contourf(X, Y, model_vortex_reshaped, cmap='jet', levels=50, norm=norm_model_vortex, vmin=model_vortex_min, vmax=model_vortex_max)
        axs[1, i].set_title(f'{name} Vortex')
    
    # Plot true values
    true_out_reshaped = np.linalg.norm(out.reshape(nx, ny, 3), axis=2)
    true_vortex_reshaped = vortex.reshape(nx, ny)
    axs[0, -1].contourf(X, Y, true_out_reshaped, cmap='jet', levels=50, norm=norm_model_out, vmin=model_out_min, vmax=model_out_max)
    axs[0, -1].set_title('True OUTL')
    axs[1, -1].contourf(X, Y, true_vortex_reshaped, cmap='jet', levels=50, norm=norm_model_vortex, vmin=model_vortex_min, vmax=model_vortex_max)
    axs[1, -1].set_title('True Vortex')
    
    fig.colorbar(cm.ScalarMappable(norm=norm_model_out, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
    fig.colorbar(cm.ScalarMappable(norm=norm_model_vortex, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.savefig(f'{plot_path}/values.png')
    plt.close()
    

def plot_modules_difference(model, model_name, pts, out, vortex, plot_path, nx, ny, dv=1e-3):
    
    pts, out, vortex, X, Y = generate_grid_data(nx, ny, return_mesh=True)
    
    model_out = model.forward(torch.from_numpy(pts).float().to(model.device)).cpu().detach().numpy() # (N, 5)
    div_net_out = model_out[:,3:]
    mom_net_out = model_out[:,:2]
    
    div_err = np.abs(out[:, :2] - div_net_out)
    mom_err = np.abs(out[:, :2] - mom_net_out)
    rel_dist = np.linalg.norm((div_net_out - mom_net_out) / (np.linalg.norm(out[:, :2], axis=1)[:,None] + 1e-8), axis=1)
    smape = (2 * np.linalg.norm(div_net_out - mom_net_out, axis=1) / (np.linalg.norm(div_net_out) + np.linalg.norm(mom_net_out)))
    module_dist = np.linalg.norm(div_net_out - mom_net_out, axis=1)
    # Save in a dictionary
    errors = {
        'div_err': div_err,
        'mom_err': mom_err
    }
    values = {
        'out': div_net_out,
        'vortex': mom_net_out
    }


    with open(f'{plot_path}/module_distance_{model_name}.txt', 'w') as f:
        div_err_norm = np.linalg.norm(errors['div_err'], ord=2)*dv
        mom_err_norm = np.linalg.norm(errors['mom_err'], ord=2)*dv
        distance_norm = np.linalg.norm(module_dist, ord=2)*dv
        div_err_max = np.max(errors['div_err'])
        mom_err_max = np.max(errors['mom_err'])
        distance_max = np.max(module_dist)
        f.write(f"{model_name}:\n")
        f.write(f"DivNet Error Norm: {div_err_norm}\n")
        f.write(f"MomNet Error Norm: {mom_err_norm}\n")
        f.write(f"Module Distance Norm: {distance_norm}\n")
        f.write(f"Maximum DivNet Error: {div_err_max}\n")
        f.write(f"Maximum MomNet Error: {mom_err_max}\n")
        f.write(f"Maximum Module Distance: {distance_max}\n\n")
        f.write(f"Average Relative Distance: {100*np.mean(rel_dist)}\n")
        f.write(f"Maximum Relative Distance: {100*np.max(rel_dist)}\n")
        f.write(f"Normalized Module Distance Norm: {distance_norm / (np.linalg.norm(out[:, :2], ord=2)*dv + 1e-8)}\n")
        f.write(f"Average smape: {smape.mean()}\n")
        f.write(f"Maximum smape: {smape.max()}\n")
        
    
def plot_models_consistencies(models, model_names, pts, out, vortex, plot_path, nx, ny, dv=1e-3):
    consistencies_models = {}
    
    pts, out, vortex, X, Y = generate_grid_data(nx, ny, return_mesh=True)
    
    for model, name in zip(models, model_names):
        # Calculate model consistency
        mom_cons, div_cons = model.evaluate_pde_residuals(torch.from_numpy(pts).float().to(model.device))
        mom_cons = mom_cons.cpu().detach().numpy()
        div_cons = div_cons.cpu().detach().numpy()
        # Save in a dictionary
        consistencies = {
            'mom_cons': mom_cons,
            'div_cons': div_cons
        }
        
        consistencies_models[name] = consistencies
    
    with open(f'{plot_path}/model_consistencies.txt', 'w') as f:
        for name, consistencies in consistencies_models.items():
            mom_cons_norm = np.linalg.norm(consistencies['mom_cons'], ord=2)*dv
            div_cons_norm = np.linalg.norm(consistencies['div_cons'], ord=2)*dv
            f.write(f"{name}:\n")
            f.write(f"Momentum Consistency Norm: {mom_cons_norm}\n")
            f.write(f"Divergence Consistency Norm: {div_cons_norm}\n\n")
    
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, len(models), figsize=(20, 10))
    
    # Determine the min and max values for the color scales
    mom_cons_min = np.inf
    mom_cons_max = -np.inf
    div_cons_min = np.inf
    div_cons_max = -np.inf
    
    for name in model_names:
        mom_cons = consistencies_models[name]['mom_cons']
        div_cons = consistencies_models[name]['div_cons']
        mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
        div_cons_reshaped = div_cons.reshape(nx, ny)
        
        mom_cons_min = min(mom_cons_min, mom_cons_reshaped.min())
        mom_cons_max = max(mom_cons_max, mom_cons_reshaped.max())
        if name != 'NCL':
            div_cons_min = min(div_cons_min, div_cons_reshaped.min())
            div_cons_max = max(div_cons_max, div_cons_reshaped.max())
    
    norm_mom_cons = plt.Normalize(vmin=mom_cons_min, vmax=mom_cons_max)
    norm_div_cons = plt.Normalize(vmin=div_cons_min, vmax=div_cons_max)
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        mom_cons = consistencies_models[name]['mom_cons']
        div_cons = consistencies_models[name]['div_cons']
        mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
        div_cons_reshaped = div_cons.reshape(nx, ny)
        
        # Plot momentum consistency
        mom_cons_plot = axs[0, i].contourf(X, Y, mom_cons_reshaped, cmap='jet', levels=50, norm=norm_mom_cons, vmin=mom_cons_min, vmax=mom_cons_max)
        axs[0, i].set_title(f'{name} Momentum Consistency')
        
        # Plot divergence consistency
        if name == 'NCL':
            axs[1, i].axis('off')
            axs[1, i].set_title(f'{name} Divergence Consistency')
        else:
            div_cons_plot = axs[1, i].contourf(X, Y, div_cons_reshaped, cmap='jet', levels=50, norm=norm_div_cons, vmin=div_cons_min, vmax=div_cons_max)
            axs[1, i].set_title(f'{name} Divergence Consistency')
    
    # Add a single colorbar for all subplots
    from matplotlib import cm
    fig.colorbar(cm.ScalarMappable(norm=norm_mom_cons, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
    if 'div_cons_plot' in locals():
        fig.colorbar(cm.ScalarMappable(norm=norm_div_cons, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.savefig(f'{plot_path}/consistencies.png')
    plt.close()
    

from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter1d

def plot_individual_loss_curves(losses, model_name, plot_path):
    div_loss_list = gaussian_filter1d(losses['div_losses'], sigma=2)
    mom_loss_list = gaussian_filter1d(losses['mom_losses'], sigma=2)
    out_loss_list = gaussian_filter1d(losses['sol_losses'], sigma=2)
    bc_loss_list = gaussian_filter1d(losses['bc_losses'], sigma=2)
    align_loss_list = gaussian_filter1d(losses['align_losses'], sigma=2)
    step_list = np.arange(len(div_loss_list))*100
    
    # Plot the losses
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(step_list, div_loss_list, label='Incompressibility eqn. Loss')
    plt.plot(step_list, mom_loss_list, label='Momentum eqn. Loss')
    plt.plot(step_list, out_loss_list, label='Prediction Error')
    plt.plot(step_list, align_loss_list, label='Alignment Loss')
    plt.title(f'{model_name} Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    
    plt.savefig(f'{plot_path}/test_losses.png')
    plt.close()
    
    
    
def plot_double_loss_curves(losses, model_names, plot_path):
    # Plot the losses
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2, figsize=(12,6), layout='compressed', sharey=True)
    
    for i in range(len(model_names)):
        curr_losses = losses[i]
        div_loss_list = curr_losses['div_losses']
        mom_loss_list = curr_losses['mom_losses']
        out_loss_list = curr_losses['sol_losses']
        bc_loss_list = curr_losses['bc_losses']
        align_loss_list = curr_losses['align_losses']
        step_list = np.arange(len(div_loss_list))
        if len(step_list) < 10000:
            step_list = step_list*100
        def smooth(data, window_size=100):
            if len(data)>1000:
                window_size=1000
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        axs[i].plot(step_list[:len(smooth(mom_loss_list))], smooth(mom_loss_list), label='(KF.M) residual')
        axs[i].plot(step_list[:len(smooth(div_loss_list))], smooth(div_loss_list), label='(KF.I) residual')
        axs[i].plot(step_list[:len(smooth(align_loss_list))], smooth(align_loss_list), label='Alignment Loss')
        axs[i].plot(step_list[:len(smooth(out_loss_list))], smooth(out_loss_list), label='Prediction Error')
        axs[i].set_title(f'{model_names[i]} Losses')
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].grid()
    
    plt.savefig(f'{plot_path}/test_losses.png')
    plt.close()
