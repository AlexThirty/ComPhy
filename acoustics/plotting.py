import numpy as np
import torch
from torch.func import vmap, jacrev
# Load the data from the file
points = np.load('data/points_full.npy')
data = np.load('data/data_full.npy')
T = np.load('data/t.npy')
X = np.load('data/X.npy')
Y = np.load('data/Y.npy')
from matplotlib import cm

print(T.shape)
print(X.shape)
print(Y.shape)
print(points.shape)

print(data.shape)

data = data.reshape((T.shape[0], X.shape[0], Y.shape[0], 3))

def plot_single_times(model, model_name, t_list, plot_path):
    import matplotlib.pyplot as plt

    pts = np.load('data/points_full.npy')
    out = np.load('data/data_full.npy')
    T = np.load('data/t.npy')
    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')

    nx, ny = X.shape[0], Y.shape[0]

    batch_size = 10000
    model_pressure_list = []
    model_vel_list = []

    for i in range(0, len(pts), batch_size):
        batch_pts = torch.from_numpy(pts[i:i+batch_size]).float().to(model.device)
        model_out_batch = model.forward(batch_pts).cpu().detach().numpy()
        model_pressure_list.append(model_out_batch[:, 0])
        model_vel_list.append(model_out_batch[:, 1:])

    model_pressure = np.concatenate(model_pressure_list, axis=0)
    model_vel = np.concatenate(model_vel_list, axis=0)

    fig, axs = plt.subplots(len(t_list), 3, figsize=(18, 6 * len(t_list)))
    pressure_min = min(out[:, 0].min(), model_pressure.min())
    pressure_max = max(out[:, 0].max(), model_pressure.max())
    norm_pressure = plt.Normalize(vmin=pressure_min, vmax=pressure_max)

    for row, t in enumerate(t_list):
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)

        true_pressure = out[time_mask, 0].reshape(nx, ny)
        pred_pressure = model_pressure[time_mask].reshape(nx, ny)
        err_pressure = np.abs(pred_pressure - true_pressure)

        # Ground truth
        im_true = axs[row, 0].contourf(X, Y, true_pressure, cmap='jet', levels=50, norm=norm_pressure)
        axs[row, 0].set_title(f'True Pressure at t={t}')
        # Prediction
        im_pred = axs[row, 1].contourf(X, Y, pred_pressure, cmap='jet', levels=50, norm=norm_pressure)
        axs[row, 1].set_title(f'Predicted Pressure at t={t}')
        # Error
        err_norm = plt.Normalize(vmin=0, vmax=err_pressure.max())
        im_err = axs[row, 2].contourf(X, Y, err_pressure, cmap='jet', levels=50, norm=err_norm)
        axs[row, 2].set_title(f'Error at t={t}')

    # Add colorbars below the subplots

    # Colorbar for true/pred columns (shared)
    cbar_ax1 = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    fig.colorbar(cm.ScalarMappable(norm=norm_pressure, cmap='jet'), cax=cbar_ax1, orientation='horizontal', label='Pressure')
    cbar_ax1.xaxis.set_ticks_position('bottom')

    # Colorbar for error column
    cbar_ax2 = fig.add_axes([0.80, 0.08, 0.15, 0.02])
    fig.colorbar(cm.ScalarMappable(norm=err_norm, cmap='jet'), cax=cbar_ax2, orientation='horizontal', label='Error')
    cbar_ax2.xaxis.set_ticks_position('bottom')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(f'{plot_path}')
    plt.close()



def plot_models_errors(models, model_names, t_list, plot_path):


    pts = np.load('data/points_full.npy')
    out = np.load('data/data_full.npy')
    T = np.load('data/t.npy')
    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')
    
    nx, ny = X.shape[0], Y.shape[0]
    nt = T.shape[0]
    
    dx = (X.max() - X.min()) / nx
    dy = (Y.max() - Y.min()) / ny
    dt = (T.max() - T.min()) / nt
    
    dv = dx * dy * dt
    
    
    
    errors_models = {}
    for model, name in zip(models, model_names):
        # Calculate model error
        batch_size = 10000
        model_pressure_list = []
        model_vel_list = []
        
        for i in range(0, len(pts), batch_size):
            batch_pts = torch.from_numpy(pts[i:i+batch_size]).float().to(model.device)
            model_out_batch = model.forward(batch_pts, return_final=True).cpu().detach().numpy()
            model_pressure_list.append(model_out_batch[:, 0])
            model_vel_list.append(model_out_batch[:, 1:])
        
        model_pressure = np.concatenate(model_pressure_list, axis=0)
        model_vel = np.concatenate(model_vel_list, axis=0)
        
        pressure_err = np.abs(model_pressure - out[:, 0])
        vel_err = np.abs(model_vel - out[:, 1:])
        # Calculate vorticity error
        # Save in a dictionary
        errors = {
            'pressure_err': pressure_err,
            'vel_err': vel_err,
            'model_pressure': model_pressure,
            'model_vel': model_vel,
            'out_err': np.concatenate([pressure_err[:, None], vel_err], axis=1)
        }
        
        errors_models[name] = errors
        
        with open(f'{plot_path}/model_errors.txt', 'w') as f:
            for name, errors in errors_models.items():
                pressure_err_norm = np.linalg.norm(errors['pressure_err'], ord=2)*dv
                vel_err_norm = np.linalg.norm(errors['vel_err'], ord=2)*dv
                tot_err_norm = np.linalg.norm(np.concatenate([errors['pressure_err'][:,None], errors['vel_err']], axis=1), ord=2)*dv
                max_pressure_err = np.max(errors['pressure_err'])
                max_vel_err = np.max(np.linalg.norm(errors['vel_err'], axis=1))
                max_tot_err = np.max(np.linalg.norm(errors['out_err'], axis=1))
                f.write(f"{name}:\n")
                f.write(f"Pressure Error Norm: {pressure_err_norm}\n")
                f.write(f"Velocity Error Norm: {vel_err_norm}\n")
                f.write(f"Total Error Norm: {tot_err_norm}\n")
                f.write(f"Max Pressure Error: {max_pressure_err}\n")
                f.write(f"Max Velocity Error: {max_vel_err}\n")
                f.write(f"Max Total Error: {max_tot_err}\n\n")
            
    from matplotlib import pyplot as plt
        # Plot the errors and values for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(2, len(models), figsize=(25, 10))
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        pressure_err_min = np.inf
        pressure_err_max = -np.inf
        vel_err_min = np.inf
        vel_err_max = -np.inf
        
        for name in model_names:
            pressure_err = errors_models[name]['pressure_err'][time_mask]
            vel_err = errors_models[name]['vel_err'][time_mask]
            pressure_err_reshaped = pressure_err.reshape(nx, ny)
            vel_err_reshaped = np.linalg.norm(vel_err.reshape(nx, ny, 2), axis=2)
            
            pressure_err_min = min(pressure_err_min, pressure_err_reshaped.min())
            pressure_err_max = max(pressure_err_max, pressure_err_reshaped.max())
            vel_err_min = min(vel_err_min, vel_err_reshaped.min())
            vel_err_max = max(vel_err_max, vel_err_reshaped.max())
        
        norm_pressure_err = plt.Normalize(vmin=pressure_err_min, vmax=pressure_err_max)
        norm_vel_err = plt.Normalize(vmin=vel_err_min, vmax=vel_err_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            pressure_err = errors_models[name]['pressure_err'][time_mask]
            vel_err = errors_models[name]['vel_err'][time_mask]
            pressure_err_reshaped = pressure_err.reshape(nx, ny)
            vel_err_reshaped = np.linalg.norm(vel_err.reshape(nx, ny, 2), axis=2)
            
            # Plot pressure error
            pressure_err_plot = axs[0, i].contourf(X, Y, pressure_err_reshaped, cmap='jet', levels=50, norm=norm_pressure_err, vmin=pressure_err_min, vmax=pressure_err_max)
            axs[0, i].set_title(f'{name} Pressure Error at t={t}')
            
            # Plot velocity error
            vel_err_plot = axs[1, i].contourf(X, Y, vel_err_reshaped, cmap='jet', levels=50, norm=norm_vel_err, vmin=vel_err_min, vmax=vel_err_max)
            axs[1, i].set_title(f'{name} Velocity Error at t={t}')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        fig.colorbar(cm.ScalarMappable(norm=norm_pressure_err, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_vel_err, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/errors_t_{t}.png')
        plt.close()
        
        # Plot overall error norm for each model in a multi-column layout
        fig, axs = plt.subplots(1, len(models), figsize=(4*len(models)+2, 4), layout='compressed', sharex=True, sharey=True)
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
        fig, axs = plt.subplots(2, len(models) + 1, figsize=(25, 10))
        # Determine the min and max values for the color scales
        model_pressure_min = np.inf
        model_pressure_max = -np.inf
        model_vel_min = np.inf
        model_vel_max = -np.inf
        
        for name in model_names:
            model_pressure = errors_models[name]['model_pressure'][time_mask]
            model_vel = errors_models[name]['model_vel'][time_mask]
            model_pressure_reshaped = model_pressure.reshape(nx, ny)
            model_vel_reshaped = model_vel.reshape(nx, ny, 2)
            
            model_pressure_min = min(model_pressure_min, model_pressure_reshaped.min())
            model_pressure_max = max(model_pressure_max, model_pressure_reshaped.max())
            model_vel_min = min(model_vel_min, np.linalg.norm(model_vel_reshaped, axis=2).min())
            model_vel_max = max(model_vel_max, np.linalg.norm(model_vel_reshaped, axis=2).max())
        
        norm_model_pressure = plt.Normalize(vmin=model_pressure_min, vmax=model_pressure_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            # Plot model pressure
            model_pressure = errors_models[name]['model_pressure'][time_mask]
            model_pressure_reshaped = model_pressure.reshape(nx, ny)
            model_pressure_plot = axs[0, i + 1].contourf(X, Y, model_pressure_reshaped, cmap='jet', levels=50, norm=norm_model_pressure, vmin=model_pressure_min, vmax=model_pressure_max)
            axs[0, i + 1].set_title(f'{name} Model Pressure at t={t}')
            
            # Plot model velocity streamplot
            model_vel = errors_models[name]['model_vel'][time_mask]
            model_vel_reshaped = model_vel.reshape(nx, ny, 2)
            axs[1, i + 1].streamplot(Y, X, model_vel_reshaped[:, :, 1], model_vel_reshaped[:, :, 0], color=np.linalg.norm(model_vel_reshaped, axis=2), cmap='jet', norm=plt.Normalize(vmin=model_vel_min, vmax=model_vel_max))
            axs[1, i + 1].set_title(f'{name} Model Velocity at t={t}')
        
        # Plot true values
        true_pressure = out[time_mask, 0]
        true_pressure_reshaped = true_pressure.reshape(nx, ny)
        true_pressure_plot = axs[0, 0].contourf(X, Y, true_pressure_reshaped, cmap='jet', levels=50, norm=norm_model_pressure, vmin=model_pressure_min, vmax=model_pressure_max)
        axs[0, 0].set_title(f'True Pressure at t={t}')
        
        true_vel = out[time_mask, 1:]
        true_vel_reshaped = true_vel.reshape(nx, ny, 2)
        axs[1, 0].streamplot(Y, X, true_vel_reshaped[:, :, 1], true_vel_reshaped[:, :, 0], color=np.linalg.norm(true_vel_reshaped, axis=2), cmap='jet', norm=plt.Normalize(vmin=model_vel_min, vmax=model_vel_max))
        axs[1, 0].set_title(f'True Velocity at t={t}')
        
        # Add a single colorbar for all subplots
        fig.colorbar(cm.ScalarMappable(norm=norm_model_pressure, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin=model_vel_min, vmax=model_vel_max), cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        plt.savefig(f'{plot_path}/values_t_{t}.png')
        plt.close()


def plot_models_consistencies(models, model_names, t_list, plot_path):
    pts = np.load('data/points_full.npy')
    out = np.load('data/data_full.npy')
    T = np.load('data/t.npy')
    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')
    
    nx, ny = X.shape[0], Y.shape[0]
    nt = T.shape[0]
    
    dx = (X.max() - X.min()) / nx
    dy = (Y.max() - Y.min()) / ny
    dt = (T.max() - T.min()) / nt
    
    dv = dx * dy * dt
    print('dx:', dx)
    print('dy:', dy)
    print('dt:', dt)
    print('dv:', dv)

    
    consistencies_models = {}
    for model, name in zip(models, model_names):
        # Calculate model consistency in batches
        batch_size = 10000
        pressure_cons_list = []
        velx_cons_list = []
        vely_cons_list = []
        
        for i in range(0, len(pts), batch_size):
            batch_pts = torch.from_numpy(pts[i:i+batch_size]).float().to(model.device)
            pressure_cons_batch, velx_cons_batch, vely_cons_batch = model.evaluate_consistency(batch_pts)
            pressure_cons_list.append(pressure_cons_batch.cpu().detach().numpy())
            velx_cons_list.append(velx_cons_batch.cpu().detach().numpy())
            vely_cons_list.append(vely_cons_batch.cpu().detach().numpy())
        
        pressure_cons = np.concatenate(pressure_cons_list, axis=0)
        velx_cons = np.concatenate(velx_cons_list, axis=0)
        vely_cons = np.concatenate(vely_cons_list, axis=0)
        # Save in a dictionary
        consistencies = {
            'pressure_cons': pressure_cons,
            'velx_cons': velx_cons,
            'vely_cons': vely_cons
        }
        
        consistencies_models[name] = consistencies
        
        with open(f'{plot_path}/model_consistencies.txt', 'w') as f:
            for name, consistencies in consistencies_models.items():
                pressure_cons_norm = np.linalg.norm(consistencies['pressure_cons'], ord=2)*dv
                velx_cons_norm = np.linalg.norm(consistencies['velx_cons'], ord=2)*dv
                vely_cons_norm = np.linalg.norm(consistencies['vely_cons'], ord=2)*dv
                f.write(f"{name}:\n")
                f.write(f"Pressure Consistency Norm: {pressure_cons_norm}\n")
                f.write(f"Velocity X Consistency Norm: {velx_cons_norm}\n")
                f.write(f"Velocity Y Consistency Norm: {vely_cons_norm}\n\n")
        
        from matplotlib import pyplot as plt
        # Plot the consistencies for each t in t_list
    for t in t_list:
        fig, axs = plt.subplots(3, len(models), figsize=(20, 15))
        # Filter the points and outputs for the current time step
        closest_t = pts[np.argmin(np.abs(pts[:, 0] - t)), 0]
        time_mask = (pts[:, 0] == closest_t)
        
        # Determine the min and max values for the color scales
        pressure_cons_min = np.inf
        pressure_cons_max = -np.inf
        velx_cons_min = np.inf
        velx_cons_max = -np.inf
        vely_cons_min = np.inf
        vely_cons_max = -np.inf
        
        for name in model_names:
            pressure_cons = consistencies_models[name]['pressure_cons'][time_mask]
            velx_cons = consistencies_models[name]['velx_cons'][time_mask]
            vely_cons = consistencies_models[name]['vely_cons'][time_mask]
            pressure_cons_reshaped = pressure_cons.reshape(nx, ny)
            velx_cons_reshaped = velx_cons.reshape(nx, ny)
            vely_cons_reshaped = vely_cons.reshape(nx, ny)
            
            pressure_cons_min = min(pressure_cons_min, pressure_cons_reshaped.min())
            pressure_cons_max = max(pressure_cons_max, pressure_cons_reshaped.max())
            velx_cons_min = min(velx_cons_min, velx_cons_reshaped.min())
            velx_cons_max = max(velx_cons_max, velx_cons_reshaped.max())
            vely_cons_min = min(vely_cons_min, vely_cons_reshaped.min())
            vely_cons_max = max(vely_cons_max, vely_cons_reshaped.max())
        
        norm_pressure_cons = plt.Normalize(vmin=pressure_cons_min, vmax=pressure_cons_max)
        norm_velx_cons = plt.Normalize(vmin=velx_cons_min, vmax=velx_cons_max)
        norm_vely_cons = plt.Normalize(vmin=vely_cons_min, vmax=vely_cons_max)
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            pressure_cons = consistencies_models[name]['pressure_cons'][time_mask]
            velx_cons = consistencies_models[name]['velx_cons'][time_mask]
            vely_cons = consistencies_models[name]['vely_cons'][time_mask]
            pressure_cons_reshaped = pressure_cons.reshape(nx, ny)
            velx_cons_reshaped = velx_cons.reshape(nx, ny)
            vely_cons_reshaped = vely_cons.reshape(nx, ny)
            
            # Plot pressure consistency
            pressure_cons_plot = axs[0, i].contourf(X, Y, pressure_cons_reshaped, cmap='jet', levels=50, norm=norm_pressure_cons, vmin=pressure_cons_min, vmax=pressure_cons_max)
            axs[0, i].set_title(f'{name} Pressure Consistency at t={t}')
            
            # Plot velocity x consistency
            velx_cons_plot = axs[1, i].contourf(X, Y, velx_cons_reshaped, cmap='jet', levels=50, norm=norm_velx_cons, vmin=velx_cons_min, vmax=velx_cons_max)
            axs[1, i].set_title(f'{name} Velocity X Consistency at t={t}')
            
            # Plot velocity y consistency
            vely_cons_plot = axs[2, i].contourf(X, Y, vely_cons_reshaped, cmap='jet', levels=50, norm=norm_vely_cons, vmin=vely_cons_min, vmax=vely_cons_max)
            axs[2, i].set_title(f'{name} Velocity Y Consistency at t={t}')
        
        # Add a single colorbar for all subplots
        from matplotlib import cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig.colorbar(cm.ScalarMappable(norm=norm_pressure_cons, cmap='jet'), ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_velx_cons, cmap='jet'), ax=axs[1, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.colorbar(cm.ScalarMappable(norm=norm_vely_cons, cmap='jet'), ax=axs[2, :], orientation='vertical', fraction=0.02, pad=0.04)
        
        plt.savefig(f'{plot_path}/consistencies_t_{t}.png')
        plt.close()

    
    
def plot_individual_loss_curves(losses, model_name, plot_path):
    pres_loss_list = losses['pres_losses']
    velx_loss_list = losses['velx_losses']
    vely_loss_list = losses['vely_losses']
    init_loss_list = losses['ic_losses']
    bc_loss_list = losses['bc_losses']
    align_loss_list = losses['align_losses']
    tot_loss_list = losses['tot_losses']
    out_loss_list = losses['y_losses']
    step_list = np.arange(len(pres_loss_list))
    # Smoothen the losses using a moving average
    def smooth(data, window_size=1000):
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
        curr_losses = losses[i]
        pres_loss_list = curr_losses['pres_losses']
        velx_loss_list = curr_losses['velx_losses']
        vely_loss_list = curr_losses['vely_losses']
        init_loss_list = curr_losses['ic_losses']
        bc_loss_list = curr_losses['bc_losses']
        align_loss_list = curr_losses['align_losses']
        tot_loss_list = curr_losses['tot_losses']
        out_loss_list = curr_losses['y_losses']
        step_list = np.arange(len(pres_loss_list))
        
        # Smoothen the losses using a moving average
        def smooth(data, window_size=1000):
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