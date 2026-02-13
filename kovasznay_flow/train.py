import wandb
import torch
from utils import set_seed, get_model, get_data, generate_random_points, update_weights
from params import *
import os
import pickle

def train_step(
    model:torch.nn.Module,
    x_pde:torch.Tensor,
    x_bc:torch.Tensor,
    y_bc:torch.Tensor,
    optimizer:torch.optim.Optimizer,
    device:torch.device
):
    model.train()
    optimizer.zero_grad()
    loss = model.loss_fn(x_pde.to(device), x_bc.to(device), y_bc.to(device))
    loss.backward()
    optimizer.step()
    
def eval_step(
    model:torch.nn.Module,
    x_pde:torch.Tensor,
    y_pde:torch.Tensor,
    x_bc:torch.Tensor,
    y_bc:torch.Tensor,
    device:torch.device
):
    model.eval()
    with torch.no_grad():
        div_loss , mom_loss, sol_loss, bc_loss, align_loss, tot_loss = model.eval_losses(x_pde, y_pde, x_bc, y_bc)
    return div_loss, mom_loss, sol_loss, bc_loss, align_loss, tot_loss

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
    sol_dataset, bc_dataset = get_data()

    print(f"Postprocessed datasets")
    print(f"Train dataset size: {len(sol_dataset)}, train_dataset.x.shape: {sol_dataset[:][0].shape}, train_dataset.y.shape: {sol_dataset[:][1].shape}")
    batch_size = train_config["batch_size"]
    # Construct the loaders
    sol_loader = torch.utils.data.DataLoader(sol_dataset, batch_size=batch_size, shuffle=True)
    bc_loader = torch.utils.data.DataLoader(bc_dataset, batch_size=batch_size, shuffle=False)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.get("learning_rate", 1e-3))
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=train_config.get("gamma", 0.9)
    )
    
    
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

    indices = torch.randperm(sol_dataset[:][0].shape[0])
    x_sol = sol_dataset[:][0][indices].float().to(device)
    y_sol = sol_dataset[:][1][indices].float().to(device)
    # Initialize WandB
    with wandb.init(
        project="ComPhy-KovasznayFlow",
        config=config,
        name=exp_name,
        tags=[model_name, alignment_mode, weight_mode],
    ) as run:
        model.to(device)
        
        div_losses = []
        mom_losses = []
        sol_losses = []
        bc_losses = []
        align_losses = []
        tot_losses = []

        for step in range(train_steps):
            if alignment_mode == "resample":
                # Sample a dense set of grid points
                dx = 0.001
                x = torch.arange(x_min, x_max + dx, dx).to(device).float()
                y = torch.arange(y_min, y_max + dx, dx).to(device).float()
                X, Y = torch.meshgrid(x, y, indexing='xy')
                grid_points = torch.column_stack([X.flatten(), Y.flatten()]).to(device)
                grid_points.requires_grad_(True)
                # Evaluate the residual on the grid points
                with torch.no_grad():
                    mom_residual, div_residual = model.evaluate_pde_residuals(grid_points)
                mom_residual = torch.linalg.norm(mom_residual, dim=1)
                # Select the points with highest residuals
                top_mom_indices = torch.topk(mom_residual.abs(), k=batch_size//4).indices
                top_mom_points = grid_points[top_mom_indices]
                top_div_indices = torch.topk(div_residual.abs(), k=batch_size//4).indices
                top_div_points = grid_points[top_div_indices]
                x_pde = generate_random_points(batch_size//2).to(device)
                # Combine the points
                x_pde = torch.cat([top_mom_points, top_div_points, x_pde], dim=0).detach().to(device)
            else:
                x_pde = generate_random_points(batch_size).to(device)
            # Train the model
            import time
            model.train()
            start_time = time.time()
            train_step(model, x_pde, x_bc, y_bc, optimizer, device)
            train_time = time.time() - start_time
            if step % print_every == 0:
                print(f"Training time: {train_time:.4f} seconds")
            if step % print_every == 0:    
                # Evaluate the model
                model.eval()
                
                with torch.no_grad():
                    start_time = time.time()
                    losses = eval_step(model, x_sol, y_sol, x_bc, y_bc, device)
                    eval_time = time.time() - start_time
                    if step % print_every == 0:
                        print(f"Evaluation time: {eval_time:.4f} seconds")

                div_loss = losses[1].detach().cpu()
                mom_loss = losses[0].detach().cpu()
                sol_loss = losses[2].detach().cpu()
                bc_loss = losses[3].detach().cpu()
                align_loss = losses[4].detach().cpu()
                tot_loss = losses[5].detach().cpu()

                div_losses.append(div_loss)
                mom_losses.append(mom_loss)
                sol_losses.append(sol_loss)
                bc_losses.append(bc_loss)
                align_losses.append(align_loss)
                tot_losses.append(tot_loss)
                
                print(f"Step {step}, div_loss: {div_loss}, mom_loss: {mom_loss}, sol_loss: {sol_loss}, bc_loss: {bc_loss}, align_loss: {align_loss}, tot_loss: {tot_loss}")
            if step % decay_every == 0:
                # Decay the learning rate
                scheduler.step()
            if step % update_every == 0 and step > 0:
                # Update the model
                update_weights(
                    model,
                    x_pde,
                    x_bc,
                    y_bc,
                    optimizer,
                    net_prefixes,
                    objectives,
                    grad_values,
                    hessian_singular_values,
                    eigvals,
                    f"results/{exp_name}",
                    step,
                    weight_mode,
                    alpha if weight_mode != 'static' else 1
                )
                run.log({"Eigenvalues": wandb.Image(f"results/{exp_name}/eigvals/eigvals_step_{step}.png")})
                run.log({"Gradient Histograms": wandb.Image(f"results/{exp_name}/grad_hists/grad_hist_agg_step_{step}.png")})
                run.log({"Gradient on Weights": wandb.Image(f"results/{exp_name}/grad_hists/grad_hist_weights_step_{step}.png")})


            # Log the losses to WandB
            run.log({
                "div_loss": div_loss,
                "mom_loss": mom_loss,
                "sol_loss": sol_loss,
                "bc_loss": bc_loss,
                "align_loss": align_loss,
                "tot_loss": tot_loss
            })
        
        run.log({
            "final_div_loss": div_loss,
            "final_mom_loss": mom_loss,
            "final_sol_loss": sol_loss,
            "final_bc_loss": bc_loss,
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
            "div_losses": div_losses,
            "mom_losses": mom_losses,
            "sol_losses": sol_losses,
            "bc_losses": bc_losses,
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
        
