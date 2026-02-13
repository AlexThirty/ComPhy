import os
import random
import numpy as np
import torch
from models.dualpinn import DualPINN
from models.pinn import PINN
from models.ncl import NCL
from models.pinnncl import PINN_Ncl
from params import *

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def get_model(model_config:dict):
    model_name = model_config["model_name"]
    if model_name == "MP-2xPINN":
        return DualPINN(**model_config)
    elif model_name == "MP-PINN+NCL":
        return PINN_Ncl(**model_config)
    elif model_name == "PINN":
        return PINN(**model_config)
    elif model_name == "NCL":
        return NCL(**model_config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_data():
    try:
        sol_dataset = torch.load('data/sol.pt', weights_only=False)
        bc_dataset = torch.load('data/boundary_condition.pt', weights_only=False)
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}, run the generate.py script.")
        
    return sol_dataset, bc_dataset


def generate_random_points(n_points):
    pts = torch.rand(n_points, 2)
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min
    return pts


# Compute gradients for each loss component and their norms
def grad_collect(loss_fn, optimizer, model, retain_graph, **kwargs):
    optimizer.zero_grad()
    loss = loss_fn(**kwargs)
    loss = loss.sum() if loss.ndim > 0 else loss
    loss.backward(retain_graph=retain_graph)
    grads_per_layer = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_per_layer[name] = param.grad.detach().cpu().numpy().copy()
        else:
            grads_per_layer[name] = None
            
    return grads_per_layer


def collect_grad_by_name(model, func, net_prefix, grad_dict, optimizer, retain_graph=True, **kwargs):
    grads_per_layer = grad_collect(func, optimizer, model, retain_graph=retain_graph, **kwargs)
    for name, grad in grads_per_layer.items():
        if name.startswith(net_prefix) and grad is not None:
            if name not in grad_dict:
                grad_dict[name] = list(grad.flatten())
            else:
                grad_dict[name].extend(grad.flatten())

def eigvals_from_output(output, net_params):
    grads = []
    for i in range(output.shape[0]):
        if output[i].ndim == 0:
            grad = torch.autograd.grad(
                outputs=output[i],
                inputs=net_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )
            grads.append(grad)
        else:
            # If output[i] has more than one component, compute grad for each component
            for j in range(output[i].numel()):
                grad = torch.autograd.grad(
                    outputs=output[i].flatten()[j],
                    inputs=net_params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True
                )
                grads.append(grad)
    grads_flat = [np.concatenate([g.flatten().detach().cpu().numpy() for g in grad if g is not None]) for grad in grads]
    grads_mat = np.array(grads_flat)
    K = grads_mat @ grads_mat.T
    eigs = np.maximum(np.linalg.eigvalsh(K), 0)
    return np.sort(eigs)[::-1]


def plot_eigvals(eigvals:dict, save_dir:str, step:int):
    import matplotlib.pyplot as plt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Determine objectives from eigvals dict
    objectives = list(next(iter(eigvals.values())).keys()) if eigvals else []
    fig, axs = plt.subplots(len(eigvals), len(objectives), figsize=(4 * len(objectives), 3 * len(eigvals)), squeeze=False)
    for i, net_prefix in enumerate(eigvals.keys()):
        for j, objective in enumerate(objectives):
            vals = eigvals[net_prefix].get(objective, [])[-1]
            vals = np.array(vals)
            axs[i][j].plot(range(len(vals)), vals, marker='o')
            axs[i][j].set_title(f"{net_prefix} - {objective}")
            axs[i][j].set_xlabel("Index")
            axs[i][j].set_ylabel("Eigenvalue")
            axs[i][j].set_ylim(bottom=1e-6)  # Ensure y-axis starts at 0
            axs[i][j].set_yscale("log")
            axs[i][j].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'eigvals_step_{step}.png'))
    plt.close()

def plot_grad_hists(grad_values, save_dir:str, step:int):
    import matplotlib.pyplot as plt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import matplotlib.pyplot as plt

    # Collect all gradient values to determine global x and y limits
    all_grads = []
    for losses_dict in grad_values.values():
        for layer_dict in losses_dict.values():
            for layer, grads in layer_dict.items():
                if '4' in layer:  # Skip layers containing '4'
                    continue
                if grads:
                    all_grads.extend(grads[:12000])
    if all_grads:
        y_min, y_max = 1, max(np.histogram(all_grads, bins=100)[0])  # Avoid log(0)
        x_min, x_max = min(all_grads), max(all_grads)
    else:
        y_min, y_max, x_min, x_max = 1, 10, -1, 1  # Fallback values
        
    # For each net_prefix, only plot layers that start with that net_prefix
    net_layers = {net_prefix: sorted([layer for layer in {layer for loss_dict in losses_dict.values() for layer in loss_dict}
                                      if layer.startswith(net_prefix)])
                  for net_prefix, losses_dict in grad_values.items()}
    num_nets = len(grad_values)
    num_layers_per_net = [len(layers) for layers in net_layers.values()]
    max_layers = max(num_layers_per_net)
    bins = np.linspace(x_min, x_max, 100)

    # Define colors for each loss
    loss_colors = {
        "bc": "tab:red",
        "mom": "tab:blue",
        "div": "tab:green",
        "alignment": "tab:purple"
        # Add more loss names and colors as needed
    }
    fig, axs = plt.subplots(num_nets, max_layers//2-1, figsize=(2 * max_layers, 3 * num_nets), squeeze=False, sharex=True, sharey=True)
    all_plotted_losses = set()
    for row, (net_prefix, losses_dict) in enumerate(grad_values.items()):
        layers = net_layers[net_prefix]
        for col, layer in enumerate(layers):
            if '4' in layer:  # Skip layers containing '4'
                continue
            for loss_name, layer_dict in losses_dict.items():
                # Skip layers that contain 'bias' in their name
                if "bias" in layer:
                    continue
                grads_losses = np.array(layer_dict.get(layer, []))
                color = loss_colors.get(loss_name, None)
                if grads_losses.size > 0:
                    # Only plot if values are not all close to zero
                    if not np.all(np.abs(grads_losses[:12800]) < 1e-8):
                        # Plot histogram
                        axs[row, col//2].hist(grads_losses[:12800], bins=bins, alpha=0.3, label=loss_name, color=color)
                        # Overlay thin line on top of bins
                        counts, bin_edges = np.histogram(grads_losses[:12800], bins=bins)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        axs[row, col//2].plot(bin_centers, counts, '-', color=color, linewidth=0.7)
                        all_plotted_losses.add(loss_name)
            if row == 0:
                axs[row, col//2].set_title(f"Layer {col//2}")
            if row==num_nets-1 and (col//2==0 or col//2==max_layers-2) :
                axs[row, col//2].set_xlabel("Gradient Value")
            # Only set ylabel on the leftmost subplot and yscale on the leftmost and bottommost
            if col//2 == 0:
                axs[row, col//2].set_ylabel("Frequency")
            axs[row, col//2].set_yscale("log")
            axs[row, col//2].set_xlim(x_min, x_max)
            axs[row, col//2].grid(True)
        # Hide unused subplots in this row
        for col in range(len(layers), max_layers):
            fig.delaxes(axs[row, col])
        # Add a global title for the row with the module name
        #fig.suptitle(f"{net_prefix.replace('_net', ' module')}", x=0.5, y=1.02, fontsize=14)
    # Add a single legend for all subplots
    handles = [plt.Line2D([0], [0], color=loss_colors.get(loss, None), lw=4, label=loss) for loss in all_plotted_losses]
    if handles:
        fig.legend(handles, [loss for loss in all_plotted_losses], loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=len(all_plotted_losses))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'grad_hist_step_{step}.png'))
    plt.close()

    # Additional plot: aggregate gradients across all layers for each net/loss
    bins = np.linspace(x_min, x_max, 100)
    fig2, axs2 = plt.subplots(num_nets, 1, figsize=(5, 3 * num_nets), squeeze=False)
    for row, (net_prefix, losses_dict) in enumerate(grad_values.items()):
        for loss_name, layer_dict in losses_dict.items():
            grads = []
            for layer_grads in layer_dict.values():
                if layer_grads:
                    grads.extend(layer_grads)
            if grads:
                color = loss_colors.get(loss_name, None)
                axs2[row, 0].hist(grads, bins=bins, alpha=0.5, label=loss_name, color=color)
                #counts, bin_edges = np.histogram(grads, bins=bins)
                #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                #axs2[row, 0].plot(bin_centers, counts, '-', color=color)
        axs2[row, 0].set_title(f"{net_prefix.replace('_', ' ')}")
        axs2[row, 0].set_xlabel("Gradient Value")
        axs2[row, 0].set_ylabel("Frequency")
        axs2[row, 0].set_yscale("log")
        #axs2[row, 0].set_ylim(y_min, y_max)
        axs2[row, 0].set_xlim(x_min, x_max)
        axs2[row, 0].grid(True)
        axs2[row, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'grad_hist_agg_step_{step}.png'))
    plt.close()
    
    
    # Additional plot: aggregate gradients across all layers for each net/loss (weights only, no biases)
    bins = np.linspace(x_min, x_max, 100)
    fig2, axs2 = plt.subplots(num_nets, 1, figsize=(5, 3 * num_nets), squeeze=False)
    for row, (net_prefix, losses_dict) in enumerate(grad_values.items()):
        for loss_name, layer_dict in losses_dict.items():
            grads = []
            for layer_name, layer_grads in layer_dict.items():
                
                # Only collect gradients from weights (exclude biases)
                if layer_grads and ("weight" in layer_name):
                    grads.extend(layer_grads)
            if grads:
                color = loss_colors.get(loss_name, None)
                axs2[row, 0].hist(grads, bins=bins, alpha=0.5, label=loss_name, color=color)
                #counts, bin_edges = np.histogram(grads, bins=bins)
                #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                #axs2[row, 0].plot(bin_centers, counts, '-', color=color)
        axs2[row, 0].set_title(f"{net_prefix} (all layers, weights only)")
        axs2[row, 0].set_xlabel("Gradient Value")
        axs2[row, 0].set_ylabel("Frequency")
        axs2[row, 0].set_yscale("log")
        #axs2[row, 0].set_ylim(y_min, y_max)
        axs2[row, 0].set_xlim(x_min, x_max)
        axs2[row, 0].grid(True)
        axs2[row, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'grad_hist_weights_step_{step}.png'))
    plt.close()


def hvp(loss, params, v):
    # First backward: grad(loss, params)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    # Only keep coordinates of v corresponding to not None grads
    grads_not_none = [g for g in grads if g is not None]
    v_filtered = torch.cat([
        v[start:start + g.numel()]
        for g, start in zip(grads_not_none, 
            np.cumsum([0] + [g.numel() for g in grads if g is not None][:-1]))
    ])
    flat_grads = torch.cat([g.reshape(-1) for g in grads_not_none])
    # Second backward: grad(flat_grads @ v_filtered, params)
    grad_v = torch.dot(flat_grads, v_filtered)
    hv = torch.autograd.grad(grad_v, params, retain_graph=True, allow_unused=True)
    return torch.cat([h.reshape(-1) for h in hv if h is not None])

def update_weights(
    model:torch.nn.Module,
    x_pde:torch.Tensor,
    x_bc:torch.Tensor,
    y_bc:torch.Tensor,
    optimizer:torch.optim.Optimizer,
    net_prefixes:list,
    losses:list,
    grad_values:dict,
    hessian_singular_values:dict,
    eigvals:dict,
    save_dir:str,
    step:int,
    weight_type:str,
    alpha:float
):
    
    # Here we collect the gradients per layer
    for net_prefix in net_prefixes:
        if net_prefix not in grad_values:
            grad_values[net_prefix] = {}
        for loss in losses:
            if loss not in grad_values[net_prefix]:
                grad_values[net_prefix][loss] = {}
            collect_grad_by_name(model, getattr(model, f'calc_{loss}_loss'), net_prefix, grad_values[net_prefix][loss], optimizer, x_pde=x_pde, x_bc=x_bc, y_bc=y_bc, retain_graph=True)

    _ = grad_collect(getattr(model, 'calc_bc_loss'), optimizer, model, retain_graph=False, x_bc=x_bc, y_bc=y_bc)


    plot_grad_hists(grad_values, save_dir=f'{save_dir}/grad_hists', step=step)

    for net_prefix in net_prefixes:
        if net_prefix not in eigvals:
            eigvals[net_prefix] = {}
        for objective in losses:
            if objective not in eigvals[net_prefix]:
                eigvals[net_prefix][objective] = []
            output = getattr(model, f'calc_{objective}')(x_pde=x_pde[:100], x_bc=x_bc[:100], y_bc=y_bc[:100])
            net_params = list(getattr(model, net_prefix).parameters())
            eigvals_calc = eigvals_from_output(output, net_params)
            try:
                eigvals[net_prefix][objective].append(eigvals_calc)
            except KeyError:
                eigvals[net_prefix][objective] = [eigvals_calc]
    plot_eigvals(eigvals, save_dir=f'{save_dir}/eigvals', step=step)
    
    if weight_type == 'ntk':
        # Compute the trace of the last eigenvalue matrices for each net/objective
        traces = {}
        for objective in losses:
            traces[objective] = 0
            for net_prefix in net_prefixes:
                eig_list = eigvals[net_prefix].get(objective, [])
                if eig_list:
                    traces[objective] += np.sum(eig_list[-1])

        all_trace = np.sum([trace for trace in traces.values() if trace is not None])

        new_weights = {objective: all_trace/traces[objective] for objective, trace in traces.items() if trace is not None}
    elif weight_type == 'grad':
        def grad_norm_reweight(loss_fn, retain_graph=False, **kwargs):
            optimizer.zero_grad()
            loss = loss_fn(**kwargs)
            loss = loss.sum() if loss.ndim > 0 else loss
            loss.backward(retain_graph=retain_graph)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
            return norm.detach()

        grad_norms = {objective: grad_norm_reweight(getattr(model, f'calc_{objective}_loss'), retain_graph=True, x_pde=x_pde[:100], x_bc=x_bc[:100], y_bc=y_bc[:100]) for objective in losses}
        _ = grad_norm_reweight(getattr(model, f'calc_bc_loss'), retain_graph=False, x_bc=x_bc, y_bc=y_bc)
        loss_tot = sum(grad_norms.values())
        
        new_weights = {objective: loss_tot / grad_norm for objective, grad_norm in grad_norms.items() if grad_norm != 0}
        
    elif weight_type == 'static':
        new_weights = {objective: 1.0 for objective in losses}

    for weight, weight_val in new_weights.items():
        setattr(model, f'{weight}_weight', alpha*getattr(model, f'{weight}_weight') + (1-alpha) * weight_val)
        print(f"Updated {weight} weight: {getattr(model, f'{weight}_weight')}")
        
    # Power iteration
    def largest_singular_value(loss, params, device, iters=20):
        n_params = sum(p.numel() for p in params)
        v = torch.randn(n_params).to(device)
        v = v / v.norm()

        for _ in range(iters):
            Hv = hvp(loss, params, v)
            norm_Hv = Hv.norm()
            v = Hv / norm_Hv  # normalize
        return norm_Hv.item()
    
    # Compute largest singular value of Hessian for each net in DualPINN
    # Store results in a dict keyed by net_prefix

    all_internal_x = x_pde
    all_boundary_x = x_bc
    all_boundary_y = y_bc

    for net_prefix in net_prefixes:
        net_attr = getattr(model, net_prefix, None)
        if net_attr is not None:
            if net_prefix not in hessian_singular_values:
                hessian_singular_values[net_prefix] = []
            params = [p for p in net_attr.parameters() if p.requires_grad]
            loss = model.loss_fn(x_pde=all_internal_x, x_bc=all_boundary_x, y_bc=all_boundary_y)
            sigma_max = largest_singular_value(loss, params, device=next(model.parameters()).device)
            hessian_singular_values[net_prefix].append(sigma_max)
