import numpy as np
from torch.func import vmap, jacrev
from matplotlib import patches
from matplotlib import pyplot as plt

import torch





def plot_models_errors(models, model_names, t_list, plot_path):
    sol = np.load('data/solution.npy')
    nt = 7
    nx = 100
    ny = 100
    X = sol[:nx*ny, 1].reshape(nx, ny)
    Y = sol[:nx*ny, 2].reshape(nx, ny)
    # Reorder pts based on the values and sol accordingly
    pts = sol[:, :3]
    out = sol[:, 3:6]
    # Substitute NaN values with the closest point from pts
    dv = 1/nx * 1/ny * 1/nt
    # Fill NaNs in out with the closest value
    from scipy.interpolate import griddata
    print('data loaded')
    errors_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error
        batch_size = 10000
        model_out = []
        for i in range(0, len(pts), batch_size):
            print(f'Calculating error for {name} at index {i}')
            batch_pts = torch.from_numpy(pts[i:i+batch_size]).float().to(model.device)
            batch_out = model.forward(batch_pts, return_final=True).cpu().detach().numpy()
            model_out.append(batch_out)
        
        model_out = np.concatenate(model_out, axis=0)[:,:3]
        model_rho = model_out[:, 0]
        model_vel = model_out[:, 1:3]
        
        out_err = np.abs(out - model_out[:,:3])
        rho_err = np.abs(out[:, 0] - model_rho)
        vel_err = np.abs(out[:, 1:] - model_vel)
        # Calculate vorticity error
        # Save in a dictionary
        errors = {
            'out_err': out_err,
            'rho_err': rho_err,
            'vel_err': vel_err,
            'model_out': model_out,
            'model_rho': model_rho,
            'model_vel': model_vel
        }
        
        errors_models[name] = errors
    print('Errors calculated')
    
    with open(f'{plot_path}/model_errors.txt', 'w') as f:
        for name, errors in errors_models.items():
            out_err_norm = np.linalg.norm(np.nan_to_num(errors['out_err']), ord=2) * dv
            rho_err_norm = np.linalg.norm(np.nan_to_num(errors['rho_err']), ord=2) * dv
            vel_err_norm = np.linalg.norm(np.nan_to_num(errors['vel_err']), ord=2) * dv
            out_err_max = np.nanmax(np.abs(errors['out_err']))
            rho_err_max = np.nanmax(np.abs(errors['rho_err']))
            vel_err_max = np.nanmax(np.linalg.norm(errors['vel_err'], axis=1))
            f.write(f"{name}:\n")
            f.write(f"Output Error Norm: {out_err_norm}\n")
            f.write(f"Rho Error Norm: {rho_err_norm}\n")
            f.write(f"Velocity Error Norm: {vel_err_norm}\n")
            f.write(f"Maximum Output Error: {out_err_max}\n")
            f.write(f"Maximum Rho Error: {rho_err_max}\n")
            f.write(f"Maximum Velocity Error: {vel_err_max}\n\n")
    print('Errors saved')
    from matplotlib import pyplot as plt
    # Plot the errors and values for each t in t_list
    for t in t_list:
        print(f'Plotting for t={t}')
        fig, axs = plt.subplots(2, len(models), figsize=(20, 10))
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        rho_err_min = np.inf
        rho_err_max = -np.inf
        vel_err_min = np.inf
        vel_err_max = -np.inf
        
        for name in model_names:
            rho_err = errors_models[name]['rho_err'][time_mask]
            vel_err = errors_models[name]['vel_err'][time_mask]
            rho_err_reshaped = rho_err.reshape(nx, ny)
            vel_err_reshaped = np.linalg.norm(vel_err.reshape(nx, ny, 2), axis=2)
            # Handle NaNs by ignoring them in min/max calculations
            rho_err_min = min(rho_err_min, np.nanmin(rho_err_reshaped))
            rho_err_max = max(rho_err_max, np.nanmax(rho_err_reshaped))
            vel_err_min = min(vel_err_min, np.nanmin(vel_err_reshaped))
            vel_err_max = max(vel_err_max, np.nanmax(vel_err_reshaped))
            
        norm_rho_err = plt.Normalize(vmin=rho_err_min, vmax=rho_err_max)
        norm_vel_err = plt.Normalize(vmin=vel_err_min, vmax=vel_err_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            #rho_err = errors_models[name]['rho_err'][time_mask]
            rho_err = np.abs(errors_models[name]['model_rho'][time_mask] - out[time_mask, 0])
            vel_err = errors_models[name]['vel_err'][time_mask]
            rho_err_reshaped = rho_err.reshape(nx, ny)
            vel_err_reshaped = np.linalg.norm(vel_err.reshape(nx, ny, 2), axis=2)
            
            # Plot the rho err
            rho_err_plot = axs[0, i].contourf(X, Y, rho_err_reshaped, cmap='jet', levels=50, norm=norm_rho_err, vmin=rho_err_min, vmax=rho_err_max)
            axs[0, i].set_title(f'{name} Rho Error at t={t}')
            
            # Plot the velocity error
            vel_err_plot = axs[1, i].contourf(X, Y, vel_err_reshaped, cmap='jet', levels=50, norm=norm_vel_err, vmin=vel_err_min, vmax=vel_err_max)
            axs[1, i].set_title(f'{name} Velocity Error at t={t}')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        from scipy.interpolate import griddata

        fig.colorbar(cm.ScalarMappable(norm=norm_rho_err, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_vel_err, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/errors_t_{t}.png')
        plt.close()

        # Plot overall error norm for each model in a multi-column layout
        fig, axs = plt.subplots(1, len(models), figsize=(25, 5), layout='compressed', sharex=True, sharey=True)
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
        fig, axs = plt.subplots(2, len(models) + 1, figsize=(20, 8))
        # Determine the min and max values for the color scales
        model_rho_min = np.inf
        model_rho_max = -np.inf
        model_vel_min = np.inf
        model_vel_max = -np.inf
        
        for name in model_names:
            model_rho = errors_models[name]['model_rho'][time_mask]
            model_vel = errors_models[name]['model_vel'][time_mask]
            model_rho_reshaped = model_rho.reshape(nx, ny)
            model_vel_reshaped = np.linalg.norm(model_vel.reshape(nx, ny, 2), axis=2)
            
            # Handle NaNs by ignoring them in min/max calculations
            model_rho_min = min(model_rho_min, np.nanmin(model_rho_reshaped))
            model_rho_max = max(model_rho_max, np.nanmax(model_rho_reshaped))
            model_vel_min = min(model_vel_min, np.nanmin(model_vel_reshaped))
            model_vel_max = max(model_vel_max, np.nanmax(model_vel_reshaped))
        
        norm_model_rho = plt.Normalize(vmin=model_rho_min, vmax=model_rho_max)
        norm_model_vel = plt.Normalize(vmin=model_vel_min, vmax=model_vel_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            # Plot model rho
            model_rho = errors_models[name]['model_rho'][time_mask]
            model_rho_reshaped = model_rho.reshape(nx, ny)
            model_rho_plot = axs[0, i + 1].contourf(X, Y, model_rho_reshaped, cmap='jet', levels=50, norm=norm_model_rho, vmin=model_rho_min, vmax=model_rho_max)
            axs[0, i + 1].set_title(f'{name} Model Rho at t={t}')
            
            # Plot model velocity streamlines
            model_vel = errors_models[name]['model_vel'][time_mask].reshape(nx, ny, 2)
            model_vel_plot = axs[1, i + 1].streamplot(X, Y, model_vel[:, :, 0], model_vel[:, :, 1], color=np.linalg.norm(model_vel, axis=2), cmap='jet', norm=norm_model_vel)
            axs[1, i + 1].set_title(f'{name} Model Velocity at t={t}')
        
        # Plot true values
        true_rho_reshaped = out[time_mask, 0].reshape(nx, ny)
        true_rho_plot = axs[0, 0].contourf(X, Y, true_rho_reshaped, cmap='jet', levels=50, norm=norm_model_rho, vmin=model_rho_min, vmax=model_rho_max)
        axs[0, 0].set_title(f'True Rho at t={t}')
        
        true_vel_reshaped = out[time_mask, 1:].reshape(nx, ny, 2)
        true_vel_plot = axs[1, 0].streamplot(X, Y, true_vel_reshaped[:, :, 0], true_vel_reshaped[:, :, 1], color=np.linalg.norm(true_vel_reshaped, axis=2), cmap='jet', norm=norm_model_vel)
        axs[1, 0].set_title(f'True Velocity at t={t}')
        
        # Add a single colorbar for all subplots
        fig.colorbar(cm.ScalarMappable(norm=norm_model_rho, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_model_vel, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/values_t_{t}.png')
        plt.close()


def plot_models_consistencies(models, model_names, t_list, plot_path):
    
    sol = np.load('data/solution.npy')
    nt = 7
    nx = 100
    ny = 100
    X = sol[:nx*ny, 1].reshape(nx, ny)
    Y = sol[:nx*ny, 2].reshape(nx, ny)
    # Reorder pts based on the values and sol accordingly
    pts = sol[:, :3]
    out = sol[:, 3:6]
    # Substitute NaN values with the closest point from pts
    dv = 1/nx * 1/ny * 1/nt
    
    consistencies_models = {}
    for model, name in zip(models, model_names):
        # Calculate model consistency
        batch_size = 5000
        mom_cons_list = []
        div_cons_list = []
        inc_cons_list = []
        for i in range(0, len(pts), batch_size):
            print(f'Calculating consistency for {name} at index {i}')
            batch_pts = torch.from_numpy(pts[i:i+batch_size]).float().to(model.device)
            mom_cons, div_cons, inc_cons = model.evaluate_consistency(batch_pts)
            mom_cons_list.append(mom_cons.cpu().detach().numpy())
            div_cons_list.append(div_cons.cpu().detach().numpy())
            inc_cons_list.append(inc_cons.cpu().detach().numpy())
        mom_cons = np.concatenate(mom_cons_list, axis=0)
        div_cons = np.concatenate(div_cons_list, axis=0)
        inc_cons = np.concatenate(inc_cons_list, axis=0)
        # Save in a dictionary
        consistencies = {
            'mom_cons': mom_cons,
            'div_cons': div_cons,
            'inc_cons': inc_cons
        }
        
        consistencies_models[name] = consistencies
        
        with open(f'{plot_path}/model_consistencies.txt', 'w') as f:
            for name, consistencies in consistencies_models.items():
                mom_cons_norm = np.linalg.norm(consistencies['mom_cons'], ord=2) * dv
                div_cons_norm = np.linalg.norm(consistencies['div_cons'], ord=2) * dv
                inc_cons_norm = np.linalg.norm(consistencies['inc_cons'], ord=2) * dv
                f.write(f"{name}:\n")
                f.write(f"Momentum Consistency Norm: {mom_cons_norm}\n")
                f.write(f"Divergence Consistency Norm: {div_cons_norm}\n")
                f.write(f"Incompressibility Consistency Norm: {inc_cons_norm}\n\n")
        
        from matplotlib import pyplot as plt
        # Plot the consistencies for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(3, len(models), figsize=(20, 15))
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        mom_cons_min = np.inf
        mom_cons_max = -np.inf
        div_cons_min = np.inf
        div_cons_max = -np.inf
        inc_cons_min = np.inf
        inc_cons_max = -np.inf
        
        for name in model_names:
            mom_cons = consistencies_models[name]['mom_cons'][time_mask]
            div_cons = consistencies_models[name]['div_cons'][time_mask]
            inc_cons = consistencies_models[name]['inc_cons'][time_mask]
            mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
            div_cons_reshaped = div_cons.reshape(nx, ny)
            inc_cons_reshaped = inc_cons.reshape(nx, ny)
            
            mom_cons_min = min(mom_cons_min, mom_cons_reshaped.min())
            mom_cons_max = max(mom_cons_max, mom_cons_reshaped.max())
            div_cons_min = min(div_cons_min, div_cons_reshaped.min())
            div_cons_max = max(div_cons_max, div_cons_reshaped.max())
            inc_cons_min = min(inc_cons_min, inc_cons_reshaped.min())
            inc_cons_max = max(inc_cons_max, inc_cons_reshaped.max())
        
        norm_mom_cons = plt.Normalize(vmin=mom_cons_min, vmax=mom_cons_max)
        norm_div_cons = plt.Normalize(vmin=div_cons_min, vmax=div_cons_max)
        norm_inc_cons = plt.Normalize(vmin=inc_cons_min, vmax=inc_cons_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            mom_cons = consistencies_models[name]['mom_cons'][time_mask]
            div_cons = consistencies_models[name]['div_cons'][time_mask]
            inc_cons = consistencies_models[name]['inc_cons'][time_mask]
            mom_cons_reshaped = np.linalg.norm(mom_cons.reshape(nx, ny, 2), axis=2)
            div_cons_reshaped = div_cons.reshape(nx, ny)
            inc_cons_reshaped = inc_cons.reshape(nx, ny)
            
            # Plot momentum consistency
            mom_cons_plot = axs[0, i].contourf(X, Y, mom_cons_reshaped, cmap='jet', levels=50, norm=norm_mom_cons, vmin=mom_cons_min, vmax=mom_cons_max)
            axs[0, i].set_title(f'{name} Momentum Consistency at t={t}')
            
            # Plot divergence consistency
         
            div_cons_plot = axs[1, i].contourf(X, Y, div_cons_reshaped, cmap='jet', levels=50, norm=norm_div_cons, vmin=div_cons_min, vmax=div_cons_max)
            axs[1, i].set_title(f'{name} Divergence Consistency at t={t}')
            
            # Plot incompressibility consistency
            inc_cons_plot = axs[2, i].contourf(X, Y, inc_cons_reshaped, cmap='jet', levels=50, norm=norm_inc_cons, vmin=inc_cons_min, vmax=inc_cons_max)
            axs[2, i].set_title(f'{name} Incompressibility Consistency at t={t}')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        fig.colorbar(cm.ScalarMappable(norm=norm_mom_cons, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        if 'div_cons_plot' in locals():
            fig.colorbar(cm.ScalarMappable(norm=norm_div_cons, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_inc_cons, cmap='jet'), ax=axs[2, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/consistencies_t_{t}.png')
        plt.close()
        plt.close()
        
        
def plot_double_loss_curves(losses, model_names, plot_path):
    # Plot the losses
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3, figsize=(18,6), layout='compressed', sharey=True)
    
    for i in range(len(model_names)):
        curr_losses = losses[i]
        step_list = curr_losses[:,0]
        mom_loss_list = curr_losses[:,1]
        inc_loss_list = curr_losses[:,2]
        out_loss_list = curr_losses[:,3]
        init_loss_list = curr_losses[:,4]
        align_loss_list = curr_losses[:,5]
        
        def smooth(data, window_size=20):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        axs[i].plot(step_list[:len(smooth(mom_loss_list))], smooth(mom_loss_list), label='(EF.M) residual')
        axs[i].plot(step_list[:len(smooth(inc_loss_list))], smooth(inc_loss_list), label='(EF.I) residual')
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
