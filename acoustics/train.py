import wandb
import torch
from utils import set_seed, get_model, get_data, generate_random_points, update_weights
from params import *
import os
from plotting import plot_single_times
import pickle

def train_step(
    model:torch.nn.Module,
    x_pde:torch.Tensor,
    x_bc:torch.Tensor,
    y_bc:torch.Tensor,
    x_ic:torch.Tensor,
    y_ic:torch.Tensor,
    optimizer:torch.optim.Optimizer,
    device:torch.device
):
    model.train()
    optimizer.zero_grad()
    loss = model.loss_fn(
        x_pde=x_pde.to(device),
        x_bc=x_bc.to(device), y_bc=y_bc.to(device),
        x_ic=x_ic.to(device), y_ic=y_ic.to(device)
    )
    loss.backward()
    optimizer.step()
    
def eval_step(
    model:torch.nn.Module,
    x_pde:torch.Tensor,
    y_pde:torch.Tensor,
    x_bc:torch.Tensor,
    y_bc:torch.Tensor,
    x_ic:torch.Tensor,
    y_ic:torch.Tensor,
    device:torch.device
):
    model.eval()
    with torch.no_grad():
        pres_loss, velx_loss, vely_loss, y_loss, bc_loss, ic_loss, align_loss, tot_loss = model.eval_losses(
            x_pde=x_pde, y_pde=y_pde,
            x_bc=x_bc, y_bc=y_bc,
            x_ic=x_ic, y_ic=y_ic)
    return pres_loss, velx_loss, vely_loss, y_loss, bc_loss, ic_loss, align_loss, tot_loss

def run_experiment(seed:int, config:dict):
    # Set the seed for reproducibility
    set_seed(seed=seed)
    
    model_config = config["model_config"]
    train_config = config["train_config"]
    device = model_config["device"]
    
    # Load the model and data
    print(f"Running experiment with seed: {seed}")
    print(f"Model: {model_config['model_name']}")
    
    model = get_model(model_config=model_config)
    
    # Filter on the loss type
    sol_dataset, bc_dataset, ic_dataset = get_data()

    print(f"Postprocessed datasets")
    print(f"Solution dataset size: {len(sol_dataset)}, sol_dataset.x.shape: {sol_dataset[:][0].shape}, sol_dataset.y.shape: {sol_dataset[:][1].shape}")
    print(f"Boundary dataset size: {len(bc_dataset)}, bc_dataset.x.shape: {bc_dataset[:][0].shape}, bc_dataset.y.shape: {bc_dataset[:][1].shape}")
    print(f"Initial condition dataset size: {len(ic_dataset)}, ic_dataset.x.shape: {ic_dataset[:][0].shape}, ic_dataset.y.shape: {ic_dataset[:][1].shape}")
    batch_size = train_config["batch_size"]
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.get("learning_rate", 1e-3))
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=train_config.get("gamma", 0.9)
    )
    from itertools import cycle
    bc_loader = cycle(iter(torch.utils.data.DataLoader(bc_dataset, batch_size=batch_size, shuffle=True)))
    ic_loader = cycle(iter(torch.utils.data.DataLoader(ic_dataset, batch_size=batch_size, shuffle=True)))

    model_name = model_config["model_name"]
    alignment_mode = model_config["alignment_mode"]
    weight_mode = model_config["weight_mode"]
    exp_name = f"{model_name}_{alignment_mode}_{weight_mode}"
    
    train_steps = train_config["train_steps"]
    decay_every = train_config["decay_every"]
    update_every = train_config["update_every"]
    print_every = train_config["print_every"]
    alpha = train_config["alpha"]
    
    net_prefixes = model_config["net_prefixes"]
    objectives = model_config["loss_terms"]
    grad_values = {}
    eigvals = {}
    hessian_singular_values = {}
    
    indices = torch.randperm(bc_dataset[:][0].shape[0])
    x_bc = bc_dataset[:][0][indices].float().to(device)
    y_bc = bc_dataset[:][1][indices].float().to(device)

    indices = torch.randperm(ic_dataset[:][0].shape[0])
    x_ic = ic_dataset[:][0][indices].float().to(device)
    y_ic = ic_dataset[:][1][indices].float().to(device)

    indices = torch.randperm(sol_dataset[:][0].shape[0])
    x_sol = sol_dataset[:][0][indices[:10000]].float().to(device)
    y_sol = sol_dataset[:][1][indices[:10000]].float().to(device)
    # Initialize WandB
    with wandb.init(
        project="ComPhy-Acoustics",
        config=config,
        name=exp_name,
        tags=[model_name, alignment_mode, weight_mode],
    ) as run:
        model.to(device)
        
        pres_losses = []
        velx_losses = []
        vely_losses = []
        y_losses = []
        bc_losses = []
        ic_losses = []
        align_losses = []
        tot_losses = []

        for step in range(train_steps):
            ic = next(ic_loader)
            x_ic = ic[0].float().to(device)
            y_ic = ic[1].float().to(device)
            
            bc = next(bc_loader)
            x_bc = bc[0].float().to(device)
            y_bc = bc[1].float().to(device)

            
            if alignment_mode == "resample":
                # Sample a dense set of grid points
                dx = 0.005
                t = torch.arange(t_min, t_max + dx, dx).to(device).float()
                x = torch.arange(x_min, x_max + dx, dx).to(device).float()
                y = torch.arange(y_min, y_max + dx, dx).to(device).float()
                T, X, Y = torch.meshgrid(t, x, y, indexing='xy')
                grid_points = torch.column_stack([T.flatten(), X.flatten(), Y.flatten()]).to(device)
                grid_points.requires_grad_(True)
                # Evaluate the residual on the grid points
                with torch.no_grad():
                    pres_residual, velx_residual, vely_residual = model.evaluate_consistency(grid_points)
                # Select the points with highest residuals
                top_pres_indices = torch.topk(pres_residual.abs(), k=batch_size//8).indices
                top_pres_points = grid_points[top_pres_indices]
                top_velx_indices = torch.topk(velx_residual.abs(), k=batch_size//8).indices
                top_velx_points = grid_points[top_velx_indices]
                top_vely_indices = torch.topk(vely_residual.abs(), k=batch_size//8).indices
                top_vely_points = grid_points[top_vely_indices]
                x_pde = generate_random_points(batch_size).to(device)
                # Combine the points
                x_pde = torch.cat([top_pres_points, top_velx_points, top_vely_points, x_pde], dim=0).detach().to(device)
            else:
                x_pde = generate_random_points(batch_size).to(device)
            # Train the model
            import time
            model.train()
            start_time = time.time()
            train_step(model, x_pde, x_bc, y_bc, x_ic, y_ic, optimizer, device)
            train_time = time.time() - start_time
            if step % print_every == 0:
                print(f"Training time: {train_time:.4f} seconds")
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                losses = eval_step(model, x_sol, y_sol, x_bc, y_bc, x_ic, y_ic, device)
                eval_time = time.time() - start_time
                if step % print_every == 0:
                    print(f"Evaluation time: {eval_time:.4f} seconds")

            pres_loss = losses[0].detach().cpu()
            velx_loss = losses[1].detach().cpu()
            vely_loss = losses[2].detach().cpu()
            y_loss = losses[3].detach().cpu()
            bc_loss = losses[4].detach().cpu()
            ic_loss = losses[5].detach().cpu()
            align_loss = losses[6].detach().cpu()
            tot_loss = losses[7].detach().cpu()

            pres_losses.append(pres_loss)
            velx_losses.append(velx_loss)
            vely_losses.append(vely_loss)
            y_losses.append(y_loss)
            bc_losses.append(bc_loss)
            ic_losses.append(ic_loss)
            align_losses.append(align_loss)
            tot_losses.append(tot_loss)
            if step % decay_every == 0:
                # Decay the learning rate
                scheduler.step()
            if step % print_every == 0:
                print(f"Step {step}, pres_loss: {pres_loss}, velx_loss: {velx_loss}, vely_loss: {vely_loss}, y_loss: {y_loss}, bc_loss: {bc_loss}, ic_loss: {ic_loss}, align_loss: {align_loss}, tot_loss: {tot_loss}")

            if step % update_every == 0 and step > 0:
                # Update the model
                if model_name != "PINNsformer":
                    update_weights(
                        model,
                        x_pde,
                        x_bc,
                        y_bc,
                        x_ic,
                        y_ic,
                        optimizer,
                        net_prefixes,
                        objectives,
                        grad_values,
                        hessian_singular_values,
                        eigvals,
                        f"results/{exp_name}",
                        step,
                        weight_mode,
                        alpha 
                    )
                if not os.path.exists(f"results/{exp_name}/plots"):
                    os.makedirs(f"results/{exp_name}/plots")
                
                plot_single_times(
                    model,
                    model_name,
                    [0., 0.06, 0.12, 0.18, 0.24],
                    f'results/{exp_name}/plots/pressure_step_{step}.png'
                )

                run.log({"Eigenvalues": wandb.Image(f"results/{exp_name}/eigvals/eigvals_step_{step}.png")})
                run.log({"Gradient Histograms": wandb.Image(f"results/{exp_name}/grad_hists/aggregate_step_{step}.png")})
                run.log({"Gradient on Weights": wandb.Image(f"results/{exp_name}/grad_hists/weights_step_{step}.png")})
                run.log({"Plot": wandb.Image(f"results/{exp_name}/plots/pressure_step_{step}.png")})


            # Log the losses to WandB
            run.log({
                "pres_loss": pres_loss,
                "velx_loss": velx_loss,
                "vely_loss": vely_loss,
                "y_loss": y_loss,
                "bc_loss": bc_loss,
                "ic_loss": ic_loss,
                "align_loss": align_loss,
                "tot_loss": tot_loss
            })
        
        run.log({
            "final_pres_loss": pres_loss,
            "final_velx_loss": velx_loss,
            "final_vely_loss": vely_loss,
            "final_y_loss": y_loss,
            "final_bc_loss": bc_loss,
            "final_ic_loss": ic_loss,
            "final_align_loss": align_loss,
            "final_tot_loss": tot_loss
        })
        
        import matplotlib.pyplot as plt

        # Save grad_values
        with open(f"results/{exp_name}/grad_values.pkl", "wb") as f:
            pickle.dump(grad_values, f)

        # Save eigvals
        with open(f"results/{exp_name}/eigvals.pkl", "wb") as f:
            pickle.dump(eigvals, f)

        # Save hessian_singular_values
        with open(f"results/{exp_name}/hessian_singular_values.pkl", "wb") as f:
            pickle.dump(hessian_singular_values, f)
            
        # Save losses
        losses_dict = {
            "pres_losses": pres_losses,
            "velx_losses": velx_losses,
            "vely_losses": vely_losses,
            "y_losses": y_losses,
            "bc_losses": bc_losses,
            "ic_losses": ic_losses,
            "align_losses": align_losses,
            "tot_losses": tot_losses
        }
        with open(f"results/{exp_name}/losses.pkl", "wb") as f:
            pickle.dump(losses_dict, f)
        
        if hessian_singular_values:
            fig, axs = plt.subplots(len(hessian_singular_values), 1, figsize=(8, 4 * len(hessian_singular_values)), squeeze=False)
            for idx, (key, values) in enumerate(hessian_singular_values.items()):
                axs[idx, 0].plot(values)
                axs[idx, 0].set_title(f"Hessian Singular Values: {key}")
                axs[idx, 0].set_xlabel("Step")
                axs[idx, 0].set_ylabel("Singular Value")
            plt.tight_layout()
            plt.savefig(f"results/{exp_name}/hessian_singular_values.png")
            plt.close(fig)
            run.log({"Hessian Singular Values": wandb.Image(f"results/{exp_name}/hessian_singular_values.png")})
        
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        # Save the model checkpoint
        torch.save(model.state_dict(), f"saved_models/{exp_name}.pth")
        run.finish()
        
