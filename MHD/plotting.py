import torch
from params import x_min, x_max, y_min, y_max, t_min, t_max
import numpy as np
from torch.func import vmap, jacrev, jacfwd, hessian
nx = 128
ny = 128
nt = 21

def plot_single_times(model, model_name, t_list, save_name, nx, ny, nt):
    import matplotlib.pyplot as plt

    # Generate grid data
    sol_data = torch.load(f"data/sol.pt", weights_only=False)
    pts = sol_data[:][0].reshape((461,128,128,3))
    out = sol_data[:][1].reshape((461,128,128,6))

    pts = pts[::23]
    pts = pts.reshape(-1, 3)
    out = out[::23]
    out = out.reshape(-1, 6)

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    from matplotlib import cm
    # Model prediction
    model_out = model.forward(pts.to(model.device).float(), return_final=True).cpu().detach().numpy()[:,:6]

    fig, axs = plt.subplots(len(t_list), 3, figsize=(18, 6 * len(t_list)), sharex=True, sharey=True)

    # Precompute min/max for each column across all t_list
    true_min, true_max = np.inf, -np.inf
    pred_min, pred_max = np.inf, -np.inf
    err_min, err_max = np.inf, -np.inf

    true_out_reshaped_list = []
    pred_out_reshaped_list = []
    error_list = []

    num_vars = out.shape[1] if len(out.shape) > 1 else 1

    # Prepare to plot all variables in a single figure
    fig, axs = plt.subplots(len(t_list), 3 * num_vars, figsize=(6 * 3 * num_vars, 6 * len(t_list)), sharex=True, sharey=True)

    # Precompute min/max for each variable and column across all t_list
    minmax = []
    true_out_reshaped_all = []
    pred_out_reshaped_all = []
    error_all = []

    for var_idx in range(num_vars):
        true_min, true_max = np.inf, -np.inf
        pred_min, pred_max = np.inf, -np.inf
        err_min, err_max = np.inf, -np.inf

        true_out_reshaped_list = []
        pred_out_reshaped_list = []
        error_list = []

        for t in t_list:
            pts_flat = pts.reshape(-1, pts.shape[-1])
            closest_t = pts_flat[np.argmin(np.abs(pts_flat[:, 0] - t)), 0]
            time_mask = (np.abs(pts_flat[:, 0] - closest_t) < 1e-8)

            true_out = out[time_mask]
            true_out_reshaped = true_out[:, var_idx].reshape(nx, ny)
            true_out_reshaped_list.append(true_out_reshaped)
            true_min = min(true_min, true_out_reshaped.min())
            true_max = max(true_max, true_out_reshaped.max())

            pred_out = model_out[time_mask]
            pred_out_reshaped = pred_out[:, var_idx].reshape(nx, ny)
            pred_out_reshaped_list.append(pred_out_reshaped)
            pred_min = min(pred_min, pred_out_reshaped.min())
            pred_max = max(pred_max, pred_out_reshaped.max())

            error = np.abs(true_out_reshaped - pred_out_reshaped)
            error_list.append(error)
            err_min = min(err_min, error.min())
            err_max = max(err_max, error.max())

        minmax.append((true_min, true_max, pred_min, pred_max, err_min, err_max))
        true_out_reshaped_all.append(true_out_reshaped_list)
        pred_out_reshaped_all.append(pred_out_reshaped_list)
        error_all.append(error_list)

    for row_idx, t in enumerate(t_list):
        for var_idx in range(num_vars):
            true_min, true_max, pred_min, pred_max, err_min, err_max = minmax[var_idx]
            norm_true = plt.Normalize(vmin=true_min, vmax=true_max)
            norm_pred = plt.Normalize(vmin=pred_min, vmax=pred_max)
            norm_err = plt.Normalize(vmin=err_min, vmax=err_max)

            # True
            im_true = axs[row_idx, 3 * var_idx + 0].contourf(X, Y, true_out_reshaped_all[var_idx][row_idx], cmap='jet', levels=50, norm=norm_true)
            axs[row_idx, 3 * var_idx + 0].set_title(f'True (var {var_idx}) t={t}')
            axs[row_idx, 3 * var_idx + 0].set_xlabel('x')
            axs[row_idx, 3 * var_idx + 0].set_ylabel('y')

            # Prediction
            im_pred = axs[row_idx, 3 * var_idx + 1].contourf(X, Y, pred_out_reshaped_all[var_idx][row_idx], cmap='jet', levels=50, norm=norm_pred)
            axs[row_idx, 3 * var_idx + 1].set_title(f'{model_name} (var {var_idx}) t={t}')
            axs[row_idx, 3 * var_idx + 1].set_xlabel('x')
            axs[row_idx, 3 * var_idx + 1].set_ylabel('y')

            # Error
            im_err = axs[row_idx, 3 * var_idx + 2].contourf(X, Y, error_all[var_idx][row_idx], cmap='jet', levels=50, norm=norm_err)
            axs[row_idx, 3 * var_idx + 2].set_title(f'Error (var {var_idx}) t={t}')
            axs[row_idx, 3 * var_idx + 2].set_xlabel('x')
            axs[row_idx, 3 * var_idx + 2].set_ylabel('y')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Add colorbars below each column group for each variable
    for var_idx in range(num_vars):
        true_min, true_max, pred_min, pred_max, err_min, err_max = minmax[var_idx]
        norm_true = plt.Normalize(vmin=true_min, vmax=true_max)
        norm_pred = plt.Normalize(vmin=pred_min, vmax=pred_max)
        norm_err = plt.Normalize(vmin=err_min, vmax=err_max)
        # True
        divider = make_axes_locatable(axs[-1, 3 * var_idx + 0])
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm_true, cmap='jet'), cax=cax, orientation='horizontal')
        cb.set_label(f'True Output (var {var_idx})')
        # Prediction
        divider = make_axes_locatable(axs[-1, 3 * var_idx + 1])
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm_pred, cmap='jet'), cax=cax, orientation='horizontal')
        cb.set_label(f'Prediction (var {var_idx})')
        # Error
        divider = make_axes_locatable(axs[-1, 3 * var_idx + 2])
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm_err, cmap='jet'), cax=cax, orientation='horizontal')
        cb.set_label(f'Error (var {var_idx})')

    plt.tight_layout()
    plt.savefig(f'{save_name}')
    plt.close()
    '''
    for var_idx in range(num_vars):
        true_min, true_max = np.inf, -np.inf
        pred_min, pred_max = np.inf, -np.inf
        err_min, err_max = np.inf, -np.inf

        true_out_reshaped_list = []
        pred_out_reshaped_list = []
        error_list = []

        for t in t_list:
            pts_flat = pts.reshape(-1, pts.shape[-1])
            closest_t = pts_flat[np.argmin(np.abs(pts_flat[:, 0] - t)), 0]
            time_mask = (np.abs(pts_flat[:, 0] - closest_t) < 1e-8)

            true_out = out[time_mask]
            true_out_reshaped = true_out[:, var_idx].reshape(nx, ny)
            true_out_reshaped_list.append(true_out_reshaped)
            true_min = min(true_min, true_out_reshaped.min())
            true_max = max(true_max, true_out_reshaped.max())

            pred_out = model_out[time_mask]
            pred_out_reshaped = pred_out[:, var_idx].reshape(nx, ny)
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

        fig, axs = plt.subplots(len(t_list), 3, figsize=(18, 6 * len(t_list)), sharex=True, sharey=True)
        for row_idx, t in enumerate(t_list):
            im_true = axs[row_idx, 0].contourf(X, Y, true_out_reshaped_list[row_idx], cmap='jet', levels=50, norm=norm_true)
            axs[row_idx, 0].set_title(f'True Output (var {var_idx}) at t={t}')

            im_pred = axs[row_idx, 1].contourf(X, Y, pred_out_reshaped_list[row_idx], cmap='jet', levels=50, norm=norm_pred)
            axs[row_idx, 1].set_title(f'{model_name} Prediction (var {var_idx}) at t={t}')

            im_err = axs[row_idx, 2].contourf(X, Y, error_list[row_idx], cmap='jet', levels=50, norm=norm_err)
            axs[row_idx, 2].set_title(f'Error (var {var_idx}) at t={t}')

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
        plt.savefig(f'{save_name}_var{var_idx}.png')
        plt.close()
    
    '''
    plt.close()


def plot_models_errors(models, model_names, t_list, plot_path):


    # Generate grid data
    sol_data = torch.load(f"data/sol.pt", weights_only=False)
    pts = sol_data[:][0].reshape((461,128,128,3))
    out = sol_data[:][1].reshape((461,128,128,6))

    pts = pts[::23]
    pts = pts.reshape(-1, 3)
    out = out[::23]
    out = out.reshape(-1, 6).numpy()

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt
    
    dv = dx*dy*dt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    from matplotlib import cm
    # Model prediction    
    
    errors_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error
        batch_size = 10000
        model_density_list = []
        model_vel_list = []
        model_pressure_list = []
        model_mag_list = []
        
        for i in range(0, len(pts), batch_size):
            batch_pts = pts[i:i+batch_size].float().to(model.device)
            model_out_batch = model.forward(batch_pts, return_final=True).cpu().detach().numpy()
            # model_out_batch shape: (batch_size, 6) -> [density, vel_x, vel_y, pressure, mag_x, mag_y]
            model_density_list.append(model_out_batch[:, 0])
            model_vel_list.append(model_out_batch[:, 1:3])
            model_pressure_list.append(model_out_batch[:, 3])
            model_mag_list.append(model_out_batch[:, 4:6])
        
        model_density = np.concatenate(model_density_list, axis=0)
        model_vel = np.concatenate(model_vel_list, axis=0)
        model_pressure = np.concatenate(model_pressure_list, axis=0)
        model_mag = np.concatenate(model_mag_list, axis=0)
        
        density_err = np.abs(model_density - out[:, 0])
        vel_err = np.abs(model_vel - out[:, 1:3])
        pressure_err = np.abs(model_pressure - out[:, 3])
        mag_err = np.abs(model_mag - out[:, 4:6])
        # Save in a dictionary
        errors = {
            'density_err': density_err,
            'vel_err': vel_err,
            'pressure_err': pressure_err,
            'mag_err': mag_err,
            'model_density': model_density,
            'model_vel': model_vel,
            'model_pressure': model_pressure,
            'model_mag': model_mag,
            'out_err': np.concatenate([
                density_err[:, None], vel_err, pressure_err[:, None], mag_err
            ], axis=1)
        }
        
        errors_models[name] = errors
        
        with open(f'{plot_path}/model_errors.txt', 'w') as f:
            for name, errors in errors_models.items():
                density_err_norm = np.linalg.norm(errors['density_err'], ord=2)*dv
                vel_err_norm = np.linalg.norm(errors['vel_err'], ord=2)*dv
                pressure_err_norm = np.linalg.norm(errors['pressure_err'], ord=2)*dv
                mag_err_norm = np.linalg.norm(errors['mag_err'], ord=2)*dv
                tot_err_norm = np.linalg.norm(errors['out_err'], ord=2)*dv
                max_density_err = np.max(errors['density_err'])
                max_vel_err = np.max(np.linalg.norm(errors['vel_err'], axis=1))
                max_pressure_err = np.max(errors['pressure_err'])
                max_mag_err = np.max(np.linalg.norm(errors['mag_err'], axis=1))
                max_tot_err = np.max(np.linalg.norm(errors['out_err'], axis=1))
                f.write(f"{name}:\n")
                f.write(f"Density Error Norm: {density_err_norm}\n")
                f.write(f"Velocity Error Norm: {vel_err_norm}\n")
                f.write(f"Pressure Error Norm: {pressure_err_norm}\n")
                f.write(f"Magnetic Error Norm: {mag_err_norm}\n")
                f.write(f"Total Error Norm: {tot_err_norm}\n")
                f.write(f"Max Density Error: {max_density_err}\n")
                f.write(f"Max Velocity Error: {max_vel_err}\n")
                f.write(f"Max Pressure Error: {max_pressure_err}\n")
                f.write(f"Max Magnetic Error: {max_mag_err}\n")
                f.write(f"Max Total Error: {max_tot_err}\n\n")
            
    from matplotlib import pyplot as plt
    # Plot the errors and values for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(4, len(models), figsize=(25, 20), sharex=True, sharey=True)
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        density_err_min = np.inf
        density_err_max = -np.inf
        vel_err_min = np.inf
        vel_err_max = -np.inf
        pressure_err_min = np.inf
        pressure_err_max = -np.inf
        mag_err_min = np.inf
        mag_err_max = -np.inf
        
        for name in model_names:
            density_err = errors_models[name]['density_err'][time_mask]
            vel_err = errors_models[name]['vel_err'][time_mask]
            pressure_err = errors_models[name]['pressure_err'][time_mask]
            mag_err = errors_models[name]['mag_err'][time_mask]
            density_err_reshaped = density_err.reshape(nx, ny)
            vel_err_reshaped = np.linalg.norm(vel_err.reshape(nx, ny, 2), axis=2)
            pressure_err_reshaped = pressure_err.reshape(nx, ny)
            mag_err_reshaped = np.linalg.norm(mag_err.reshape(nx, ny, 2), axis=2)
            
            density_err_min = min(density_err_min, density_err_reshaped.min())
            density_err_max = max(density_err_max, density_err_reshaped.max())
            vel_err_min = min(vel_err_min, vel_err_reshaped.min())
            vel_err_max = max(vel_err_max, vel_err_reshaped.max())
            pressure_err_min = min(pressure_err_min, pressure_err_reshaped.min())
            pressure_err_max = max(pressure_err_max, pressure_err_reshaped.max())
            mag_err_min = min(mag_err_min, mag_err_reshaped.min())
            mag_err_max = max(mag_err_max, mag_err_reshaped.max())
        
        norm_density_err = plt.Normalize(vmin=density_err_min, vmax=density_err_max)
        norm_vel_err = plt.Normalize(vmin=vel_err_min, vmax=vel_err_max)
        norm_pressure_err = plt.Normalize(vmin=pressure_err_min, vmax=pressure_err_max)
        norm_mag_err = plt.Normalize(vmin=mag_err_min, vmax=mag_err_max)
        
        row_titles = ['Density Error', 'Velocity Error', 'Pressure Error', 'Magnetic Error']
        for row, title in enumerate(row_titles):
            axs[row, 0].set_ylabel(title)
        for col, name in enumerate(model_names):
            axs[0, col].set_title(name)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            density_err = errors_models[name]['density_err'][time_mask]
            vel_err = errors_models[name]['vel_err'][time_mask]
            pressure_err = errors_models[name]['pressure_err'][time_mask]
            mag_err = errors_models[name]['mag_err'][time_mask]
            density_err_reshaped = density_err.reshape(nx, ny)
            vel_err_reshaped = np.linalg.norm(vel_err.reshape(nx, ny, 2), axis=2)
            pressure_err_reshaped = pressure_err.reshape(nx, ny)
            mag_err_reshaped = np.linalg.norm(mag_err.reshape(nx, ny, 2), axis=2)
            
            axs[0, i].contourf(X, Y, density_err_reshaped, cmap='jet', levels=50, norm=norm_density_err)
            axs[1, i].contourf(X, Y, vel_err_reshaped, cmap='jet', levels=50, norm=norm_vel_err)
            axs[2, i].contourf(X, Y, pressure_err_reshaped, cmap='jet', levels=50, norm=norm_pressure_err)
            axs[3, i].contourf(X, Y, mag_err_reshaped, cmap='jet', levels=50, norm=norm_mag_err)
        
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlabel('x')
        
        from matplotlib import cm
        fig.colorbar(cm.ScalarMappable(norm=norm_density_err, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_vel_err, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_pressure_err, cmap='jet'), ax=axs[2, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_mag_err, cmap='jet'), ax=axs[3, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/errors_t_{t}.png')
        plt.close()
        
        # Plot overall error norm for each model in a multi-column layout
        fig, axs = plt.subplots(1, len(models), figsize=(4*len(models), 5), layout='compressed', sharex=True, sharey=True)
        overall_err_min = np.inf
        overall_err_max = -np.inf
        
        for name in model_names:
            overall_err = np.linalg.norm(errors_models[name]['out_err'][time_mask].reshape(nx, ny, 6), axis=2)
            overall_err_min = min(overall_err_min, np.nanmin(overall_err))
            overall_err_max = max(overall_err_max, np.nanmax(overall_err))
        
        norm_overall_err = plt.Normalize(vmin=overall_err_min, vmax=overall_err_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            overall_err = np.linalg.norm(errors_models[name]['out_err'][time_mask].reshape(nx, ny, 6), axis=2)
            axs[i].contourf(X, Y, overall_err, cmap='jet', levels=50, norm=norm_overall_err)
            axs[i].set_xlabel('x')
        axs[0].set_ylabel('Overall Error Norm')
        for col, name in enumerate(model_names):
            axs[col].set_title(name)
        
        fig.colorbar(cm.ScalarMappable(norm=norm_overall_err, cmap='jet'), ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        plt.savefig(f'{plot_path}/overall_error_norm_t_{t}.png')
        plt.close()
        
        # Plot the true and model values in a new figure
        fig, axs = plt.subplots(4, len(models) + 1, figsize=(30, 20), sharex=True, sharey=True)
        row_titles = ['Density', 'Velocity', 'Pressure', 'Magnetic']
        axs[0, 0].set_title('True')
        for row, title in enumerate(row_titles):
            axs[row, 0].set_ylabel(title)
        for col, name in enumerate(model_names):
            axs[0, col + 1].set_title(name)
        
        model_density_min = np.inf
        model_density_max = -np.inf
        model_vel_min = np.inf
        model_vel_max = -np.inf
        model_pressure_min = np.inf
        model_pressure_max = -np.inf
        model_mag_min = np.inf
        model_mag_max = -np.inf
        
        for name in model_names:
            model_density = errors_models[name]['model_density'][time_mask]
            model_vel = errors_models[name]['model_vel'][time_mask]
            model_pressure = errors_models[name]['model_pressure'][time_mask]
            model_mag = errors_models[name]['model_mag'][time_mask]
            model_density_reshaped = model_density.reshape(nx, ny)
            model_vel_reshaped = model_vel.reshape(nx, ny, 2)
            model_pressure_reshaped = model_pressure.reshape(nx, ny)
            model_mag_reshaped = model_mag.reshape(nx, ny, 2)
            
            model_density_min = min(model_density_min, model_density_reshaped.min())
            model_density_max = max(model_density_max, model_density_reshaped.max())
            model_vel_min = min(model_vel_min, np.linalg.norm(model_vel_reshaped, axis=2).min())
            model_vel_max = max(model_vel_max, np.linalg.norm(model_vel_reshaped, axis=2).max())
            model_pressure_min = min(model_pressure_min, model_pressure_reshaped.min())
            model_pressure_max = max(model_pressure_max, model_pressure_reshaped.max())
            model_mag_min = min(model_mag_min, np.linalg.norm(model_mag_reshaped, axis=2).min())
            model_mag_max = max(model_mag_max, np.linalg.norm(model_mag_reshaped, axis=2).max())
        
        norm_model_density = plt.Normalize(vmin=model_density_min, vmax=model_density_max)
        norm_model_pressure = plt.Normalize(vmin=model_pressure_min, vmax=model_pressure_max)
        norm_model_mag = plt.Normalize(vmin=model_mag_min, vmax=model_mag_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            model_density = errors_models[name]['model_density'][time_mask]
            model_density_reshaped = model_density.reshape(nx, ny)
            axs[0, i + 1].contourf(X, Y, model_density_reshaped, cmap='jet', levels=50, norm=norm_model_density)
            model_vel = errors_models[name]['model_vel'][time_mask]
            model_vel_reshaped = model_vel.reshape(nx, ny, 2)
            axs[1, i + 1].streamplot(Y, X, model_vel_reshaped[:, :, 1], model_vel_reshaped[:, :, 0], color=np.linalg.norm(model_vel_reshaped, axis=2), cmap='jet', norm=plt.Normalize(vmin=model_vel_min, vmax=model_vel_max))
            model_pressure = errors_models[name]['model_pressure'][time_mask]
            model_pressure_reshaped = model_pressure.reshape(nx, ny)
            axs[2, i + 1].contourf(X, Y, model_pressure_reshaped, cmap='jet', levels=50, norm=norm_model_pressure)
            model_mag = errors_models[name]['model_mag'][time_mask]
            model_mag_reshaped = model_mag.reshape(nx, ny, 2)
            axs[3, i + 1].streamplot(Y, X, model_mag_reshaped[:, :, 1], model_mag_reshaped[:, :, 0], color=np.linalg.norm(model_mag_reshaped, axis=2), cmap='jet', norm=norm_model_mag)
        
        true_density = out[time_mask, 0]
        true_density_reshaped = true_density.reshape(nx, ny)
        axs[0, 0].contourf(X, Y, true_density_reshaped, cmap='jet', levels=50, norm=norm_model_density)
        true_vel = out[time_mask, 1:3]
        true_vel_reshaped = true_vel.reshape(nx, ny, 2)
        axs[1, 0].streamplot(Y, X, true_vel_reshaped[:, :, 1], true_vel_reshaped[:, :, 0], color=np.linalg.norm(true_vel_reshaped, axis=2), cmap='jet', norm=plt.Normalize(vmin=model_vel_min, vmax=model_vel_max))
        true_pressure = out[time_mask, 3]
        true_pressure_reshaped = true_pressure.reshape(nx, ny)
        axs[2, 0].contourf(X, Y, true_pressure_reshaped, cmap='jet', levels=50, norm=norm_model_pressure)
        true_mag = out[time_mask, 4:6]
        true_mag_reshaped = true_mag.reshape(nx, ny, 2)
        axs[3, 0].streamplot(Y, X, true_mag_reshaped[:, :, 1], true_mag_reshaped[:, :, 0], color=np.linalg.norm(true_mag_reshaped, axis=2), cmap='jet', norm=norm_model_mag)
        
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlabel('x')
        
        fig.colorbar(cm.ScalarMappable(norm=norm_model_density, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin=model_vel_min, vmax=model_vel_max), cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_model_pressure, cmap='jet'), ax=axs[2, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_model_mag, cmap='jet'), ax=axs[3, :], orientation='vertical', fraction=0.02, pad=0.04)
        plt.savefig(f'{plot_path}/values_t_{t}.png')
        plt.close()


def plot_models_consistencies(models, model_names, t_list, plot_path):
    # Generate grid data
    sol_data = torch.load(f"data/sol.pt", weights_only=False)
    pts = sol_data[:][0].reshape((461,128,128,3))
    out = sol_data[:][1].reshape((461,128,128,6))

    pts = pts[::23]
    pts = pts.reshape(-1, 3)
    out = out[::23]
    out = out.reshape(-1, 6).numpy()
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dt = (t_max - t_min) / nt

    dv = dx * dy * dt
    print('dx:', dx)
    print('dy:', dy)
    print('dt:', dt)
    print('dv:', dv)
    dv = dx*dy*dt
    x = np.linspace(x_min, x_max+dx, nx)
    y = np.linspace(y_min, y_max+dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    from matplotlib import cm

    
    consistencies_models = {}
    for model, name in zip(models, model_names):
        # Calculate model consistency in batches
        batch_size = 10000
        continuity_cons_list = []
        momentum_cons_list = []      # shape (batch_size, 2)
        state_cons_list = []
        induction_cons_list = []     # shape (batch_size, 2)
        gauss_cons_list = []

        for i in range(0, len(pts), batch_size):
            batch_pts = pts[i:i+batch_size].float().to(model.device)
            # Evaluate all consistencies: continuity, momentum (x,y), state, induction (x,y), gauss
            (
                continuity_cons_batch,
                momentum_cons_batch,      # shape (batch_size, 2)
                state_cons_batch,
                induction_cons_batch,     # shape (batch_size, 2)
                gauss_cons_batch
            ) = model.evaluate_consistency(batch_pts)
            continuity_cons_list.append(continuity_cons_batch.cpu().detach().numpy())
            momentum_cons_list.append(momentum_cons_batch.cpu().detach().numpy())
            state_cons_list.append(state_cons_batch.cpu().detach().numpy())
            induction_cons_list.append(induction_cons_batch.cpu().detach().numpy())
            gauss_cons_list.append(gauss_cons_batch.cpu().detach().numpy())

        continuity_cons = np.concatenate(continuity_cons_list, axis=0)
        momentum_cons = np.concatenate(momentum_cons_list, axis=0)      # shape (N, 2)
        state_cons = np.concatenate(state_cons_list, axis=0)
        induction_cons = np.concatenate(induction_cons_list, axis=0)    # shape (N, 2)
        gauss_cons = np.concatenate(gauss_cons_list, axis=0)
        # Save in a dictionary
        consistencies = {
            'continuity_cons': continuity_cons,
            'momentum_cons': momentum_cons,
            'state_cons': state_cons,
            'induction_cons': induction_cons,
            'gauss_cons': gauss_cons
        }

        consistencies_models[name] = consistencies

    with open(f'{plot_path}/model_consistencies.txt', 'w') as f:
        for name, consistencies in consistencies_models.items():
            continuity_cons_norm = np.linalg.norm(consistencies['continuity_cons'], ord=2)*dv
            momentum_cons_norm = np.linalg.norm(consistencies['momentum_cons'], ord=2)*dv
            state_cons_norm = np.linalg.norm(consistencies['state_cons'], ord=2)*dv
            induction_cons_norm = np.linalg.norm(consistencies['induction_cons'], ord=2)*dv
            gauss_cons_norm = np.linalg.norm(consistencies['gauss_cons'], ord=2)*dv
            f.write(f"{name}:\n")
            f.write(f"Continuity Consistency Norm: {continuity_cons_norm}\n")
            f.write(f"Momentum Consistency Norm: {momentum_cons_norm}\n")
            f.write(f"State Consistency Norm: {state_cons_norm}\n")
            f.write(f"Induction Consistency Norm: {induction_cons_norm}\n")
            f.write(f"Gauss Consistency Norm: {gauss_cons_norm}\n\n")
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Plot the consistencies for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(5, len(models), figsize=(25, 25), sharex=True, sharey=True)
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)

        # Determine the min and max values for the color scales for each row
        minmax = {}
        keys = [
            'continuity_cons', 'momentum_cons', 'state_cons', 'induction_cons', 'gauss_cons'
        ]
        for key in keys:
            minmax[key] = [np.inf, -np.inf]
        for name in model_names:
            # Scalar keys
            for key in ['continuity_cons', 'state_cons', 'gauss_cons']:
                cons = consistencies_models[name][key][time_mask]
                cons_reshaped = cons.reshape(nx, ny)
                minmax[key][0] = min(minmax[key][0], cons_reshaped.min())
                minmax[key][1] = max(minmax[key][1], cons_reshaped.max())
            # Vector keys
            for key in ['momentum_cons', 'induction_cons']:
                cons = consistencies_models[name][key][time_mask]
                cons_reshaped = cons.reshape(nx, ny, 2)
                cons_norm = np.linalg.norm(cons_reshaped, axis=2)
                minmax[key][0] = min(minmax[key][0], cons_norm.min())
                minmax[key][1] = max(minmax[key][1], cons_norm.max())

        norms = {key: plt.Normalize(vmin=minmax[key][0], vmax=minmax[key][1]) for key in keys}

        row_titles = [
            'Continuity Consistency',
            'Momentum Consistency (norm)',
            'State Consistency',
            'Induction Consistency (norm)',
            'Gauss Consistency'
        ]

        for row, key in enumerate(keys):
            # Title at the start of the row
            axs[row, 0].set_ylabel(row_titles[row])
            axs[row, 0].set_title(row_titles[row], loc='left', fontsize=14, fontweight='bold')
            for col, name in enumerate(model_names):
                if row == 0:
                    axs[row, col].set_title(name)
                # Plot data
                if key in ['continuity_cons', 'state_cons', 'gauss_cons']:
                    cons = consistencies_models[name][key][time_mask]
                    cons_reshaped = cons.reshape(nx, ny)
                    im = axs[row, col].contourf(X, Y, cons_reshaped, cmap='jet', levels=50, norm=norms[key])
                else:
                    cons = consistencies_models[name][key][time_mask]
                    cons_reshaped = cons.reshape(nx, ny, 2)
                    cons_norm = np.linalg.norm(cons_reshaped, axis=2)
                    im = axs[row, col].contourf(X, Y, cons_norm, cmap='jet', levels=50, norm=norms[key])
                axs[row, col].set_xlabel('x')
                axs[row, col].set_ylabel('y')
            # Add colorbar for each row (across all columns)
            divider = make_axes_locatable(axs[row, -1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(im, cax=cax, orientation='vertical')
            cb.set_label(row_titles[row])

        plt.tight_layout()
        plt.savefig(f'{plot_path}/consistencies_t_{t}.png')
        plt.close()


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
    
    
    
def plot_individual_loss_curves(losses, model_name, plot_path):
    losses = losses.T
    step_list = losses[:,0]
    pres_loss_list = losses[:,1]
    velx_loss_list = losses[:,2]
    vely_loss_list = losses[:,3]
    init_loss_list = losses[:,4]
    bc_loss_list = losses[:,5]
    align_loss_list = losses[:,6]
    tot_loss_list = losses[:,7]
    out_loss_list = losses[:,8]
    # Smoothen the losses using a moving average
    def smooth(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Plot the losses
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    #plt.plot(step_list[len(step_list) - len(smooth(pres_loss_list)):], smooth(pres_loss_list), label='Pressure Loss')
    plt.plot(step_list[len(step_list) - len(smooth(velx_loss_list)):], smooth(velx_loss_list), label='Velocity X Loss')
    plt.plot(step_list[len(step_list) - len(smooth(vely_loss_list)):], smooth(vely_loss_list), label='Velocity Y Loss')
    plt.plot(step_list[len(step_list) - len(smooth(align_loss_list)):], smooth(align_loss_list), label='Alignment Loss')
    plt.plot(step_list[len(step_list) - len(smooth(out_loss_list)):], smooth(out_loss_list), label='Prediction Error')
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
        losses[i] = losses[i].T
        curr_losses = losses[i]
        step_list = curr_losses[:,0]
        pres_loss_list = curr_losses[:,1]
        velx_loss_list = curr_losses[:,2]
        vely_loss_list = curr_losses[:,3]
        init_loss_list = curr_losses[:,4]
        bc_loss_list = curr_losses[:,5]
        align_loss_list = curr_losses[:,6]
        tot_loss_list = curr_losses[:,7]
        out_loss_list = curr_losses[:,8]
        
        # Smoothen the losses using a moving average
        def smooth(data, window_size=10):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        #axs[i].plot(step_list[len(step_list) - len(smooth(pres_loss_list)):], smooth(pres_loss_list), label='Pressure Loss')
        axs[i].plot(step_list[len(step_list) - len(smooth(velx_loss_list)):], smooth(velx_loss_list), label='(A.Vx) residual')
        axs[i].plot(step_list[len(step_list) - len(smooth(vely_loss_list)):], smooth(vely_loss_list), label='(A.Vy) residual')
        axs[i].plot(step_list[len(step_list) - len(smooth(align_loss_list)):], smooth(align_loss_list), label='Alignment Loss')
        axs[i].plot(step_list[len(step_list) - len(smooth(out_loss_list)):], smooth(out_loss_list), label=r'Prediction Error')
        axs[i].set_title(f'{model_names[i]} Losses')
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].grid()
    
    plt.savefig(f'{plot_path}/test_losses.png')
    plt.close()