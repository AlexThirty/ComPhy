import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import time
import os
seed = 30
from itertools import cycle
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
gen = torch.Generator()
gen.manual_seed(seed)

from models.dualpinn import DualPINN
from utils import generate_random_points, generate_initial_points, true_init

parser = argparse.ArgumentParser()
parser.add_argument('--inc_weight', default=1e-2, type=float, help='Weight for the divergence loss')
parser.add_argument('--init_weight', default=30., type=float, help='Weight for the boundary condition loss')
parser.add_argument('--div_weight', default=1., type=float, help='Weight for the PINN loss')
parser.add_argument('--mom_weight', default=3e-3, type=float, help='Weight for the PINN loss')
parser.add_argument('--alignment_weight', default=1., type=float, help='Weight for the alignment loss')
parser.add_argument('--lr_init', default=5e-4, type=float, help='Starting learning rate')
parser.add_argument('--device', default='cuda:2', type=str, help='Device to use')
parser.add_argument('--name', default='dualpinn', type=str, help='Experiment name')
parser.add_argument('--train_steps', default=1000, type=int, help='Number of training steps in each epoch')
parser.add_argument('--epochs', default=600, type=int, help='Number of epochs')
parser.add_argument('--mode', default='DERL', type=str, help='Mode: -1 for PINN learning, 0 for derivative learning, 1 for output learning')
parser.add_argument('--batch_size', default=1000, type=int, help='Number of samples per step')
parser.add_argument('--layers', default=8, type=int, help='Number of layers in the network')
parser.add_argument('--units', default=256, type=int, help='Number of units per layer in the network')
parser.add_argument('--restart', default=False, type=bool, help='Use grid data', action=argparse.BooleanOptionalAction)
parser.add_argument('--weight_type', default='static', type=str, help='Type of weight calculation')

args = parser.parse_args()
inc_weight = args.inc_weight
mom_weight = args.mom_weight
init_weight = args.init_weight
div_weight = args.div_weight
lr_init = args.lr_init
device = args.device
name = args.mode
train_steps = args.train_steps
epochs = args.epochs
mode = args.mode
batch_size = args.batch_size
alignment_weight = args.alignment_weight
layers = args.layers
units = args.units
weight_type = args.weight_type

from model_params import dualpinn_params
params = dualpinn_params[mode]
alignment_weight = params['alignment_weight']
div_weight = params['div_weight']
inc_weight = params['inc_weight']
mom_weight = params['mom_weight']
init_weight = params['init_weight']
lr_init = params['lr_init']

if weight_type == 'grad':
    inc_weight = 1.
    mom_weight = 1.
    init_weight = 1.
    alignment_weight = 1.
    div_weight = 1.

# Generate the dataset
from models.params import x_min, x_max, y_min, y_max, t_min, t_max, dt
n_pts = batch_size*train_steps
internal_data = generate_random_points(n_pts, x_min, x_max, y_min, y_max, t_min, t_max)
print('Internal data shape: ', internal_data.shape)
internal_dataset = TensorDataset(torch.tensor(internal_data))

initial_data = generate_initial_points(n_pts, x_min, x_max, y_min, y_max, dt)
print('Initial data shape: ', initial_data.shape)
initial_y = true_init(initial_data)
print(initial_y.shape)
initial_dataset = TensorDataset(torch.tensor(initial_data), torch.tensor(initial_y))

solution_dataset = torch.load('data/sol_dataset.pt', weights_only=False)
print('Solution dataset shape: ', solution_dataset.tensors[0].shape)
print(solution_dataset[:10][0])
print(solution_dataset[:10][1])


# Now prepare the dataloaders
internal_loader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=True, generator=gen)
initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True, generator=gen)
solution_loader = DataLoader(solution_dataset, batch_size=batch_size, shuffle=True, generator=gen)

# Initialize the model
model = DualPINN(
    div_hidden_units=[units for _ in range(layers)],
    inc_hidden_units=[units for _ in range(layers)],
    alignment_weight=alignment_weight,
    div_weight=div_weight,
    mom_weight=mom_weight,
    inc_weight=inc_weight,
    init_weight=init_weight,
    device=device
)
model.to(device)

print(model)

step_list = []
div_losses = []
inc_losses = []
mom_losses = []
y_losses = []
init_losses = []
inc_losses = []
alignment_losses = []
tot_losses = []

step_list_test = []
div_losses_test = []
inc_losses_test = []
mom_losses_test = []
y_losses_test = []
init_losses_test = []
inc_losses_test = []
alignment_losses_test = []
tot_losses_test = []

time_test = []

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-5)


# Load the latest checkpoint if available
checkpoint_dir = 'saved_models/dualpinn_checkpoints'
start_epoch = 0
if args.restart:
    print('Restarting training from scratch.')
elif os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and mode in f]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint)))
        print(f'Loaded checkpoint: {latest_checkpoint}')
    else:
        print('No checkpoints found, starting from scratch.')
else:
    print('Checkpoint directory does not exist, starting from scratch.')

remaining_epochs = epochs - start_epoch
print(f'Resuming training from epoch {start_epoch} for {remaining_epochs} more epochs.')

epochs = remaining_epochs

alpha_weight = 0.9

# Training loop
def train_loop(epochs:int,
        internal_dataloader:DataLoader,
        initial_dataloader:DataLoader,
        solution_dataloader:DataLoader,
        print_every:int=100):
    
    # Training mode for the network
    model.train()

    
    for epoch in range(epochs):
        if epoch % 5 == 0 and epoch > 0:
            

            if weight_type == 'grad':
                def grad_norm_reweight(loss_fn, *args, retain_graph=False):
                    optimizer.zero_grad()
                    loss = loss_fn(*args)
                    loss = loss.sum() if loss.ndim > 0 else loss
                    loss.backward(retain_graph=retain_graph)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
                    return norm.detach()

                all_internal_x = internal_dataset[:batch_size][0].to(device).float().requires_grad_(True)
                all_initial_x = initial_dataset[:batch_size][0].to(device).float().requires_grad_(True)
                all_initial_y = initial_dataset[:batch_size][1].to(device).float().requires_grad_(True)

                # Retain the graph for all but the last backward call
                ic_grad_norm = grad_norm_reweight(model.calc_ic_loss, all_initial_x, all_initial_y, retain_graph=True)
                mom_grad_norm = grad_norm_reweight(model.calc_mom_loss, all_internal_x, retain_graph=True)
                inc_grad_norm = grad_norm_reweight(model.calc_inc_loss, all_internal_x, retain_graph=True)
                div_grad_norm = grad_norm_reweight(model.calc_div_loss, all_internal_x, retain_graph=True)
                align_grad_norm = grad_norm_reweight(model.calc_align_loss, all_internal_x, mode, retain_graph=False)

                loss_sums = ic_grad_norm + mom_grad_norm + div_grad_norm + align_grad_norm + inc_grad_norm

                w_bc = loss_sums / ic_grad_norm
                w_mom = loss_sums / mom_grad_norm
                w_div = loss_sums / div_grad_norm
                w_align = loss_sums / align_grad_norm
                w_inc = loss_sums / inc_grad_norm
                if epoch == 0:
                    model.init_weight = w_bc
                    model.inc_weight = w_inc
                    model.mom_weight = w_mom
                    model.div_weight = w_div
                    model.alignment_weight = w_align
                else:
                    model.inc_weight = alpha_weight*model.inc_weight + (1-alpha_weight)*w_inc
                    model.init_weight = alpha_weight*model.init_weight + (1-alpha_weight)*w_bc
                    model.mom_weight = alpha_weight*model.mom_weight + (1-alpha_weight)*w_mom
                    model.div_weight = alpha_weight*model.div_weight + (1-alpha_weight)*w_div
                    model.alignment_weight = alpha_weight*model.alignment_weight + (1-alpha_weight)*w_align
                print(f'Adaptive Weights - INIT: {model.init_weight}, INC: {model.inc_weight}, MOM: {model.mom_weight}, DIV: {model.div_weight}, ALIGN: {model.alignment_weight}')

            
        
        
        start_time = time.time()
        step_prefix = epoch*len(internal_loader)
        
        for step, (pde_data, init_data, sol_data) in enumerate(zip(internal_dataloader, cycle(initial_dataloader), cycle(solution_dataloader))):
            if step > train_steps:
                break
            # Load batches from dataloaders
            x_pde = pde_data[0].to(device).float().requires_grad_(True)

            # Boundary conditions            
            x_init = init_data[0].to(device).float().requires_grad_(True)
            y_init = init_data[1].to(device).float()
            
            x_sol = sol_data[0].to(device).float().requires_grad_(True)
            y_sol = sol_data[1].to(device).float()
                        
            # Call zero grad on optimizer
            optimizer.zero_grad()
        
            loss = model.loss_fn(
                x_pde=x_pde,
                x_init=x_init, y_init=y_init, alignment_mode=mode,
            )
            # Backward the loss, calculate gradients
            loss.backward()
            # Optimizer step
            optimizer.step()
            # Printing
            if (step_prefix+step) % print_every == 0 and step>0:
                with torch.no_grad():
                    _, div_loss, mom_loss, inc_loss, y_loss, init_loss_val, alignment_loss_val, tot_loss_val = model.eval_losses(
                        step=step_prefix+step,
                        x_pde=x_pde,
                        x_init=x_init, y_init=y_init,
                        x_sol=x_sol, y_sol=y_sol,
                        alignment_mode=mode
                    )
                        
                    step_list.append(step_prefix+step)
                    div_losses.append(div_loss.item())
                    inc_losses.append(inc_loss.item())
                    mom_losses.append(mom_loss.item())
                    y_losses.append(y_loss.item())
                    init_losses.append(init_loss_val.item())
                    alignment_losses.append(alignment_loss_val.item())
                    tot_losses.append(tot_loss_val.item())
                    
                    
                    print(f'Step: {step_prefix+step}, div loss: {div_loss}, Mom loss: {mom_loss}, inc loss: {inc_loss}, y loss: {y_loss}')
                    print(f'init loss: {init_loss_val}, alignment loss: {alignment_loss_val}, Total loss: {tot_loss_val}')
                    
        end_time = time.time()
        
        epoch_time = end_time - start_time
        print(f'Epoch: {epoch}, time: {epoch_time}')
        time_test.append(epoch_time)
        
        # Testing the model
        model.eval()
        div_loss_test = 0.
        mom_loss_test = 0.
        inc_loss_test = 0.
        y_loss_test = 0.
        init_loss_test = 0.
        alignment_loss_test = 0.
        tot_loss_test = 0.
        
        with torch.no_grad():
            for (pde_data, init_data, sol_data) in zip(internal_dataloader, cycle(initial_dataloader), cycle(solution_dataloader)):
                # Load batches from dataloaders
                x_pde = pde_data[0].to(device).float().requires_grad_(True)                #

                # Boundary conditions            
                x_init = init_data[0].to(device).float().requires_grad_(True)
                y_init = init_data[1].to(device).float()
                
                x_sol = sol_data[0].to(device).float().requires_grad_(True)
                y_sol = sol_data[1].to(device).float()
                
                _, div_loss, mom_loss, inc_loss, y_loss, init_loss_val, alignment_loss_val, tot_loss_val = model.eval_losses(
                    step=step_prefix+step,
                    x_pde=x_pde,
                    x_init=x_init, y_init=y_init,
                    x_sol=x_sol, y_sol=y_sol,
                    alignment_mode=mode
                )
                
                div_loss_test += div_loss.item()
                inc_loss_test += inc_loss.item()
                mom_loss_test += mom_loss.item()
                y_loss_test += y_loss.item()
                init_loss_test += init_loss_val.item()
                alignment_loss_test += alignment_loss_val.item()
                tot_loss_test += tot_loss_val.item()
                
            div_loss_test /= len(internal_dataloader)
            inc_loss_test /= len(internal_dataloader)
            mom_loss_test /= len(internal_dataloader)
            y_loss_test /= len(internal_dataloader)
            init_loss_test /= len(internal_dataloader)
            alignment_loss_test /= len(internal_dataloader)
            tot_loss_test /= len(internal_dataloader)
                
                
             
             
        step_list_test.append(step_prefix+step)
        div_losses_test.append(div_loss_test)
        inc_losses_test.append(inc_loss_test)
        mom_losses_test.append(mom_loss_test)
        y_losses_test.append(y_loss_test)
        init_losses_test.append(init_loss_test)
        alignment_losses_test.append(alignment_loss_test)
        tot_losses_test.append(tot_loss_test)
        
        scheduler.step(metrics=tot_loss_test)
        
        if epoch % 50 == 0:
            if not os.path.exists('saved_models/dualpinn_checkpoints'):
                os.makedirs('saved_models/dualpinn_checkpoints')
            torch.save(model.state_dict(), f'saved_models/dualpinn_checkpoints/dualpinn_{mode}_{epoch}.pt')
            print('Checkpoint saved')

        print(f'Test Div loss: {div_loss_test}, Test Mom loss: {mom_loss_test}, Test inc loss: {inc_loss_test}, Test y loss: {y_loss_test}')
        print(f'Test init loss: {init_loss_test}, Test Total loss: {tot_loss_test}')
        print('------------------------------------------------------------')    
        
train_loop(epochs, internal_loader, initial_loader, solution_loader, print_every=100)

# Save the model
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
torch.save(model.state_dict(), f'saved_models/dualpinn_{mode}_{weight_type}.pt')

save_dir = f'results_dualpinn_{weight_type}'

if not os.path.exists(f'{save_dir}'):
    os.makedirs(f'{save_dir}')

import matplotlib.pyplot as plt

step_list = np.array(step_list)
div_losses = np.array(div_losses)
inc_losses = np.array(inc_losses)
mom_losses = np.array(mom_losses)
y_losses = np.array(y_losses)
init_losses = np.array(init_losses)
tot_losses = np.array(tot_losses)

step_list_test = np.array(step_list_test)
div_losses_test = np.array(div_losses_test)
inc_losses_test = np.array(inc_losses_test)
mom_losses_test = np.array(mom_losses_test)
y_losses_test = np.array(y_losses_test)
init_losses_test = np.array(init_losses_test)
tot_losses_test = np.array(tot_losses_test)
time_test = np.array(time_test)


train_losses = np.vstack((step_list, div_losses, mom_losses, inc_losses, y_losses, init_losses, alignment_losses, tot_losses)).T
test_losses = np.vstack((step_list_test, div_losses_test, mom_losses_test, inc_losses_test, y_losses_test, init_losses_test, alignment_losses_test, tot_losses_test, time_test)).T

np.save(f'{save_dir}/{mode}_train_losses.npy', train_losses)
np.save(f'{save_dir}/{mode}_test_losses.npy', test_losses)


plt.figure()

plt.plot(step_list, div_losses, label='Divergence Loss')
plt.plot(step_list, inc_losses, label='Incergence Loss')
plt.plot(step_list, mom_losses, label='Momentum Loss')
plt.plot(step_list, y_losses, label='Y Loss')
plt.plot(step_list, init_losses, label='Init Loss')
plt.plot(step_list, alignment_losses, label='alignment Loss')
plt.plot(step_list, tot_losses, label='Total Loss')

plt.yscale('log')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(f'{save_dir}/{mode}_training_losses.png')
plt.figure(figsize=(10, 8))

plt.plot(step_list_test, div_losses_test, label='Divergence Loss Test')
plt.plot(step_list_test, inc_losses_test, label='Incergence Loss Test')
plt.plot(step_list_test, mom_losses_test, label='Momentum Loss Test')
plt.plot(step_list_test, y_losses_test, label='Y Loss Test')
plt.plot(step_list_test, init_losses_test, label='Init Loss Test')
plt.plot(step_list_test, alignment_losses_test, label='alignment Loss Test')
plt.plot(step_list_test, tot_losses_test, label='Total Loss Test')

plt.yscale('log')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Testing Losses')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(f'{save_dir}/{mode}_testing_losses.png')