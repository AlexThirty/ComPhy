from generate import u_true, v_true, p_true, stream_true, vortex_true
import torch
from params import x_min, x_max, y_min, y_max, t_min, t_max
import numpy as np
from torch.func import vmap, jacrev, jacfwd, hessian

def generate_grid_data(nx: int, ny: int, nt: int, return_mesh=False):
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    t = np.linspace(t_min, t_max+dt, nt)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    pts = np.column_stack([T.reshape((-1,1)), X.reshape((-1,1)), Y.reshape((-1,1))])
    u = u_true(torch.from_numpy(pts))
    v = v_true(torch.from_numpy(pts))
    p = p_true(torch.from_numpy(pts))
    out = torch.stack([u, v, p], dim=1)
    out = torch.stack([u, v, p], dim=1)
    stream = stream_true(torch.from_numpy(pts))
    vortex = vortex_true(torch.from_numpy(pts))
    
    stream_np = stream.detach().numpy()
    vortex_np = vortex.detach().numpy()
    out_np = out.detach().numpy()
    
    if return_mesh:
        return pts, out_np, stream_np, vortex_np, X, Y, T
    return pts, out_np, stream_np, vortex_np


def plot_module_distances(model, model_name, pts, out, plot_path, nx, ny, nt, dv=1e-3):
    pts, out, stream, vortex, X, Y, T = generate_grid_data(nx, ny, nt, return_mesh=True)
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    t = np.linspace(t_min, t_max+dt, nt)
    X, Y = np.meshgrid(x, y, indexing='ij')
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
        
        

def plot_models_errors(models, model_names, pts, out, vortex, t_list, plot_path, nx, ny, nt, dv=1e-3):
    pts, out, stream, vortex, X, Y, T = generate_grid_data(nx, ny, nt, return_mesh=True)
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    t = np.linspace(t_min, t_max+dt, nt)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    errors_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error
        model_out = model.forward(torch.from_numpy(pts).float().to(model.device), return_final=True).cpu().detach().numpy()
        out_err = np.abs(out - model_out)
        # Calculate vorticity error in batches
        batch_size = 1024
        model_vortex_list = []
        for i in range(0, len(pts), batch_size):
            batch_pts = torch.from_numpy(pts[i:i + batch_size]).float().to(model.device)
            batch_Dy = vmap(jacrev(model.forward))(batch_pts).cpu().detach().numpy()
            batch_vortex = batch_Dy[:, 1, 1] - batch_Dy[:, 0, 2]
            model_vortex_list.append(batch_vortex)
        model_vortex = np.concatenate(model_vortex_list, axis=0)
        vortex_err = np.abs(vortex - model_vortex)
        # Save in a dictionary
        errors = {
            'out_err': out_err,
            'vortex_err': vortex_err,
            'model_out': model_out,
            'model_vortex': model_vortex
        }
        
        errors_models[name] = errors
    
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
    # Plot the errors and values for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(2, len(models), figsize=(20, 10), layout='compressed', sharex=True, sharey=True)
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        out_err_min = np.inf
        out_err_max = -np.inf
        vortex_err_min = np.inf
        vortex_err_max = -np.inf
        
        for name in model_names:
            out_err = errors_models[name]['out_err'][time_mask]
            vortex_err = errors_models[name]['vortex_err'][time_mask]
            out_err_reshaped = np.linalg.norm(out_err.reshape(nx, ny, 3), axis=2)
            vortex_err_reshaped = vortex_err.reshape(nx, ny)
            
            out_err_min = min(out_err_min, out_err_reshaped.min())
            out_err_max = max(out_err_max, out_err_reshaped.max())
            vortex_err_min = min(vortex_err_min, vortex_err_reshaped.min())
            vortex_err_max = max(vortex_err_max, vortex_err_reshaped.max())
        import matplotlib.colors as mcolors
        norm_out_err = plt.Normalize(vmin=out_err_min, vmax=out_err_max)
        norm_vortex_err = plt.Normalize(vmin=vortex_err_min, vmax=vortex_err_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            out_err = errors_models[name]['out_err'][time_mask]
            vortex_err = errors_models[name]['vortex_err'][time_mask]
            out_err_reshaped = np.linalg.norm(out_err.reshape(nx, ny, 3), axis=2)
            vortex_err_reshaped = vortex_err.reshape(nx, ny)
            
            # Plot output error
            out_err_plot = axs[0, i].contourf(X, Y, out_err_reshaped, cmap='jet', levels=50, norm=norm_out_err, vmin=out_err_min, vmax=out_err_max)
            axs[0, i].set_title(f'{name} Error')
            
            # Plot vortex error
            vortex_err_plot = axs[1, i].contourf(X, Y, vortex_err_reshaped, cmap='jet', levels=50, norm=norm_vortex_err, vmin=vortex_err_min, vmax=vortex_err_max)
            axs[1, i].set_title(f'{name} Vortex Error')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        fig.colorbar(cm.ScalarMappable(norm=norm_out_err, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.02)
        fig.colorbar(cm.ScalarMappable(norm=norm_vortex_err, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.02)
        
        plt.savefig(f'{plot_path}/errors_t_{t}.png')
        plt.close()
        
        # Plot overall error norm for each model in a multi-column layout
        fig, axs = plt.subplots(1, len(models), figsize=(5*len(models), 5), layout='compressed', sharex=True, sharey=True)
        overall_err_min = np.inf
        overall_err_max = -np.inf
        
        # Determine the min and max values for the color scales
        for name in model_names:
            overall_err = np.linalg.norm(errors_models[name]['out_err'][time_mask].reshape(nx, ny, 3), axis=2)
            overall_err_min = min(overall_err_min, np.nanmin(overall_err))
            overall_err_max = max(overall_err_max, np.nanmax(overall_err))
        
        norm_overall_err = plt.Normalize(vmin=overall_err_min, vmax=overall_err_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            overall_err = np.linalg.norm(errors_models[name]['out_err'][time_mask].reshape(nx, ny, 3), axis=2)
            overall_err_plot = axs[i].contourf(X, Y, overall_err, cmap='jet', levels=50, norm=norm_overall_err, vmin=overall_err_min, vmax=overall_err_max)
            axs[i].set_title(f'{name}')
        
        # Add a single colorbar for all subplots
        fig.colorbar(cm.ScalarMappable(norm=norm_overall_err, cmap='jet'), ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        
        
        plt.savefig(f'{plot_path}/overall_error_norm_t_{t}.png')
        plt.close()
        
        # Plot the true and model values in a new figure
        fig, axs = plt.subplots(2, len(models) + 1, figsize=(20, 10), layout='compressed')
        # Determine the min and max values for the color scales
        model_out_min = np.inf
        model_out_max = -np.inf
        model_vortex_min = np.inf
        model_vortex_max = -np.inf
        
        for name in model_names:
            model_out = errors_models[name]['model_out'][time_mask]
            model_vortex = errors_models[name]['model_vortex'][time_mask]
            model_out_reshaped = np.linalg.norm(model_out.reshape(nx, ny, 3), axis=2)
            model_vortex_reshaped = model_vortex.reshape(nx, ny)
            
            model_out_min = min(model_out_min, model_out_reshaped.min())
            model_out_max = max(model_out_max, model_out_reshaped.max())
            model_vortex_min = min(model_vortex_min, model_vortex_reshaped.min())
            model_vortex_max = max(model_vortex_max, model_vortex_reshaped.max())
        
        norm_model_out = plt.Normalize(vmin=model_out_min, vmax=model_out_max)
        norm_model_vortex = plt.Normalize(vmin=model_vortex_min, vmax=model_vortex_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            # Plot model output
            model_out = errors_models[name]['model_out'][time_mask]
            model_out_reshaped = np.linalg.norm(model_out.reshape(nx, ny, 3), axis=2)
            model_out_plot = axs[0, i + 1].contourf(X, Y, model_out_reshaped, cmap='jet', levels=50, norm=norm_model_out, vmin=model_out_min, vmax=model_out_max)
            axs[0, i + 1].set_title(f'{name} Model Output at t={t}')
            
            # Plot model vortex
            model_vortex = errors_models[name]['model_vortex'][time_mask]
            model_vortex_reshaped = model_vortex.reshape(nx, ny)
            model_vortex_plot = axs[1, i + 1].contourf(X, Y, model_vortex_reshaped, cmap='jet', levels=50, norm=norm_model_vortex, vmin=model_vortex_min, vmax=model_vortex_max)
            axs[1, i + 1].set_title(f'{name} Model Vortex at t={t}')
        
        # Plot true values
        true_out = out[time_mask]
        true_out_reshaped = np.linalg.norm(true_out.reshape(nx, ny, 3), axis=2)
        true_out_plot = axs[0, 0].contourf(X, Y, true_out_reshaped, cmap='jet', levels=50, norm=norm_model_out, vmin=model_out_min, vmax=model_out_max)
        axs[0, 0].set_title(f'True Output at t={t}')
        
        true_vortex_reshaped = vortex[time_mask].reshape(nx, ny)
        true_vortex_plot = axs[1, 0].contourf(X, Y, true_vortex_reshaped, cmap='jet', levels=50, norm=norm_model_vortex, vmin=model_vortex_min, vmax=model_vortex_max)
        axs[1, 0].set_title(f'True Vortex at t={t}')
        
        # Add a single colorbar for all subplots
        fig.colorbar(cm.ScalarMappable(norm=norm_model_out, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_model_vortex, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/values_t_{t}.png')
        plt.close()


def plot_models_consistencies(models, model_names, pts, out, vortex, t_list, plot_path, nx, ny, nt, dv=1e-3):
    pts, out, stream, vortex, X, Y, T = generate_grid_data(nx, ny, nt, return_mesh=True)
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    t = np.linspace(t_min, t_max+dt, nt)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    consistencies_models = {}
    batch_size = 1024  # Define a batch size
    for model, name in zip(models, model_names):
        mom_cons_list = []
        div_cons_list = []
        for i in range(0, len(pts), batch_size):
            batch_pts = torch.from_numpy(pts[i:i + batch_size]).float().to(model.device)
            # Calculate model consistency for the batch
            mom_cons_batch, div_cons_batch = model.evaluate_pde_residuals(batch_pts) if 'module' in model.__class__.__name__.lower() else model.evaluate_pde_residuals(batch_pts)
            mom_cons_list.append(mom_cons_batch.cpu().detach().numpy())
            div_cons_list.append(div_cons_batch.cpu().detach().numpy())
        # Concatenate all batches
        mom_cons = np.concatenate(mom_cons_list, axis=0)
        div_cons = np.concatenate(div_cons_list, axis=0)
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
    '''
    from matplotlib import pyplot as plt
    # Plot the consistencies for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(2, len(models), figsize=(20, 10))
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        mom_cons_min = np.inf
        mom_cons_max = -np.inf
        div_cons_min = np.inf
        div_cons_max = -np.inf
        
        for name in model_names:
            mom_cons = consistencies_models[name]['mom_cons'][time_mask]
            div_cons = consistencies_models[name]['div_cons'][time_mask]
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
            mom_cons = consistencies_models[name]['mom_cons'][time_mask]
            div_cons = consistencies_models[name]['div_cons'][time_mask]
            mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
            div_cons_reshaped = div_cons.reshape(nx, ny)
            
            # Plot momentum consistency
            mom_cons_plot = axs[0, i].contourf(X, Y, mom_cons_reshaped, cmap='jet', levels=50, norm=norm_mom_cons, vmin=mom_cons_min, vmax=mom_cons_max)
            axs[0, i].set_title(f'{name} Momentum Consistency at t={t}')
            
            # Plot divergence consistency
            if name == 'NCL':
                axs[1, i].axis('off')
                axs[1, i].set_title(f'{name} Divergence Consistency at t={t}')
            else:
                div_cons_plot = axs[1, i].contourf(X, Y, div_cons_reshaped, cmap='jet', levels=50, norm=norm_div_cons, vmin=div_cons_min, vmax=div_cons_max)
                axs[1, i].set_title(f'{name} Divergence Consistency at t={t}')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm_mom_cons, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        #fig.colorbar(mom_cons_plot, ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        if 'div_cons_plot' in locals():
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm_div_cons, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        plt.savefig(f'{plot_path}/consistencies_t_{t}.png')
        plt.close()
    plt.close()
    '''
    
def plot_individual_loss_curves(losses, model_name, plot_path):
    step_list = losses[:,0]
    div_loss_list = losses[:,1]
    mom_loss_list = losses[:,2]
    out_loss_list = losses[:,3]
    bc_loss_list = losses[:,4]
    init_loss_list = losses[:,5]
    align_loss_list = losses[:,6]
    
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
    fig, axs = plt.subplots(1, 2, figsize=(12,6), layout='compressed', sharey=True)
    
    for i in range(len(model_names)):
        curr_losses = losses[i]
        step_list = curr_losses[:,0]
        div_loss_list = curr_losses[:,1]
        mom_loss_list = curr_losses[:,2]
        out_loss_list = curr_losses[:,3]
        bc_loss_list = curr_losses[:,4]
        init_loss_list = curr_losses[:,5]
        align_loss_list = curr_losses[:,6]
        
        def smooth(data, window_size=1):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        axs[i].plot(step_list[:len(smooth(mom_loss_list))], smooth(mom_loss_list), label='(TG.M) residual')
        axs[i].plot(step_list[:len(smooth(div_loss_list))], smooth(div_loss_list), label='(TG.I) residual')
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
    

def plot_single_times(model, model_name, t_list, save_name, nx, ny, nt):
    import matplotlib.pyplot as plt

    # Generate grid data
    pts, out, stream, vortex, X, Y, T = generate_grid_data(nx, ny, nt, return_mesh=True)

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    from matplotlib import cm
    # Model prediction
    model_out = model.forward(torch.from_numpy(pts).float().to(model.device), return_final=True).cpu().detach().numpy()

    fig, axs = plt.subplots(len(t_list), 3, figsize=(18, 6 * len(t_list)), sharex=True, sharey=True)

    # Precompute min/max for each column across all t_list
    true_min, true_max = np.inf, -np.inf
    pred_min, pred_max = np.inf, -np.inf
    err_min, err_max = np.inf, -np.inf

    true_out_reshaped_list = []
    pred_out_reshaped_list = []
    error_list = []

    for t in t_list:
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)

        true_out = out[time_mask]
        true_out_reshaped = np.linalg.norm(true_out.reshape(nx, ny, 3), axis=2)
        true_out_reshaped_list.append(true_out_reshaped)
        true_min = min(true_min, true_out_reshaped.min())
        true_max = max(true_max, true_out_reshaped.max())

        pred_out = model_out[time_mask]
        pred_out_reshaped = np.linalg.norm(pred_out.reshape(nx, ny, 3), axis=2)
        pred_out_reshaped_list.append(pred_out_reshaped)
        pred_min = min(pred_min, pred_out_reshaped.min())
        pred_max = max(pred_max, pred_out_reshaped.max())

        error = np.abs(true_out_reshaped - pred_out_reshaped)
        error_list.append(error)
        err_min = min(err_min, error.min())
        err_max = max(err_max, error.max())

    norm_true = plt.Normalize(vmin=true_min, vmax=true_max)
    norm_pred = plt.Normalize(vmin=pred_min, vmax=pred_max)
    norm_err = plt.Normalize(vmin=err_min, vmax=err_max)

    for row_idx, t in enumerate(t_list):
        im_true = axs[row_idx, 0].contourf(X, Y, true_out_reshaped_list[row_idx], cmap='jet', levels=50, norm=norm_true)
        axs[row_idx, 0].set_title(f'True Output at t={t}')

        im_pred = axs[row_idx, 1].contourf(X, Y, pred_out_reshaped_list[row_idx], cmap='jet', levels=50, norm=norm_pred)
        axs[row_idx, 1].set_title(f'{model_name} Prediction at t={t}')

        im_err = axs[row_idx, 2].contourf(X, Y, error_list[row_idx], cmap='jet', levels=50, norm=norm_err)
        axs[row_idx, 2].set_title(f'Error at t={t}')

        for col in range(3):
            axs[row_idx, col].set_xlabel('x')
            axs[row_idx, col].set_ylabel('y')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Add colorbars below each column
    for col, (norm, label) in enumerate([(norm_true, 'True OUTL'), (norm_pred, 'Prediction'), (norm_err, 'Error')]):
        divider = make_axes_locatable(axs[-1, col])
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax, orientation='horizontal')
        cb.set_label(label)

    plt.tight_layout()
    plt.savefig(f'{save_name}')
    plt.close()


def plot_models_errors_module(models, model_names, pts, out, vortex,t_list, plot_path, nx, ny, nt, dv=1e-3):
    pts, out, stream, vortex, X, Y, T = generate_grid_data(nx, ny, nt, return_mesh=True)
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    t = np.linspace(t_min, t_max+dt, nt)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    errors_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error
        model_out = model.forward(torch.from_numpy(pts).float().to(model.device)).cpu().detach().numpy()
        model_out_div = model_out[:,:2]
        model_out_mom = model_out[:,2:4]
        out_err_div = np.abs(model_out_div - out[:, :2])
        out_err_mom = np.abs(model_out_mom - out[:, :2])
        # Calculate vorticity error
        model_Dy = vmap(jacrev(model.forward))(torch.from_numpy(pts).float().to(model.device)).cpu().detach().numpy()
        model_vortex_div = model_Dy[:, 1, 1] - model_Dy[:, 0, 2]
        model_vortex_mom = model_Dy[:, 3, 1] + model_Dy[:, 2, 0]
        vortex_err_div = np.abs(model_vortex_div - vortex)
        vortex_err_mom = np.abs(model_vortex_mom - vortex)
        # Save in a dictionary
        errors = {
            'out_err_div': out_err_div,
            'out_err_mom': out_err_mom,
            'vortex_err_div': vortex_err_div,
            'vortex_err_mom': vortex_err_mom,
            'model_out_div': model_out_div,
            'model_out_mom': model_out_mom,
            'model_vortex_div': model_vortex_div,
            'model_vortex_mom': model_vortex_mom
        }
        
        errors_models[name] = errors

        with open(f'{plot_path}/model_errors.txt', 'w') as f:
            for name, errors in errors_models.items():
                out_err_div_norm = np.linalg.norm(errors['out_err_div'], ord=2) * dv
                out_err_mom_norm = np.linalg.norm(errors['out_err_mom'], ord=2) * dv
                vortex_err_div_norm = np.linalg.norm(errors['vortex_err_div'], ord=2) * dv
                vortex_err_mom_norm = np.linalg.norm(errors['vortex_err_mom'], ord=2) * dv
                out_err_div_max = np.max(errors['out_err_div'])
                out_err_mom_max = np.max(errors['out_err_mom'])
                vortex_err_div_max = np.max(errors['vortex_err_div'])
                vortex_err_mom_max = np.max(errors['vortex_err_mom'])
                f.write(f"{name}:\n")
                f.write(f"Div Output Error Norm: {out_err_div_norm}\n")
                f.write(f"Mom Output Error Norm: {out_err_mom_norm}\n")
                f.write(f"Div Vortex Error Norm: {vortex_err_div_norm}\n")
                f.write(f"Mom Vortex Error Norm: {vortex_err_mom_norm}\n")
                f.write(f"Maximum Div Output Error: {out_err_div_max}\n")
                f.write(f"Maximum Mom Output Error: {out_err_mom_max}\n")
                f.write(f"Maximum Div Vortex Error: {vortex_err_div_max}\n")
                f.write(f"Maximum Mom Vortex Error: {vortex_err_mom_max}\n\n")
    

def plot_models_consistencies_modules(models, model_names, pts, out, vortex, t_list, plot_path, nx, ny, nt, dv=1e-3):
    pts, out, stream, vortex, X, Y, T = generate_grid_data(nx, ny, nt, return_mesh=True)
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    t = np.linspace(t_min, t_max+dt, nt)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    consistencies_models = {}
    batch_size = 1024  # Define a batch size
    for model, name in zip(models, model_names):
        mom_cons_list = []
        div_cons_list = []
        for i in range(0, len(pts), batch_size):
            batch_pts = torch.from_numpy(pts[i:i + batch_size]).float().to(model.device)
            # Calculate model consistency for the batch
            mom_cons_batch, div_cons_batch = model.evaluate_consistency_module(batch_pts)
            mom_cons_list.append(mom_cons_batch.cpu().detach().numpy())
            div_cons_list.append(div_cons_batch.cpu().detach().numpy())
        # Concatenate all batches
        mom_cons = np.concatenate(mom_cons_list, axis=0)
        div_cons = np.concatenate(div_cons_list, axis=0)
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
    # Plot the consistencies for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(2, len(models), figsize=(20, 10))
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        mom_cons_min = np.inf
        mom_cons_max = -np.inf
        div_cons_min = np.inf
        div_cons_max = -np.inf
        
        for name in model_names:
            mom_cons = consistencies_models[name]['mom_cons'][time_mask]
            div_cons = consistencies_models[name]['div_cons'][time_mask]
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
            mom_cons = consistencies_models[name]['mom_cons'][time_mask]
            div_cons = consistencies_models[name]['div_cons'][time_mask]
            mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
            div_cons_reshaped = div_cons.reshape(nx, ny)
            
            # Plot momentum consistency
            mom_cons_plot = axs[0, i].contourf(X, Y, mom_cons_reshaped, cmap='jet', levels=50, norm=norm_mom_cons, vmin=mom_cons_min, vmax=mom_cons_max)
            axs[0, i].set_title(f'{name} Momentum Consistency at t={t}')
            
            # Plot divergence consistency
            if name == 'NCL':
                axs[1, i].axis('off')
                axs[1, i].set_title(f'{name} Divergence Consistency at t={t}')
            else:
                div_cons_plot = axs[1, i].contourf(X, Y, div_cons_reshaped, cmap='jet', levels=50, norm=norm_div_cons, vmin=div_cons_min, vmax=div_cons_max)
                axs[1, i].set_title(f'{name} Divergence Consistency at t={t}')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm_mom_cons, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        #fig.colorbar(mom_cons_plot, ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        if 'div_cons_plot' in locals():
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm_div_cons, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        plt.savefig(f'{plot_path}/consistencies_t_{t}.png')
        plt.close()
    plt.close()