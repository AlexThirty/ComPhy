
import torch.utils
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch.func import vmap, jacrev, jacfwd, hessian
import numpy as np

# Equation parameters
L = 1.
Re = 50  # Reynolds Number
nu = 1 / Re  # Viscosity

def generate_random_points(N:int, x_min:float, x_max:float, y_min:float, y_max:float):
    random_points = torch.rand((N, 2)) * torch.tensor([x_max-x_min, y_max-y_min]) + torch.tensor([x_min, y_min])
    return random_points

def generate_boundary_points(N:int, x_min:float, x_max:float, y_min:float, y_max:float):
    boundary_points = np.vstack([
        np.column_stack([np.linspace(x_min, x_max, N), np.full(N, y_min)]),  # Bottom boundary
        np.column_stack([np.linspace(x_min, x_max, N), np.full(N, y_max)]),   # Top boundary
        np.column_stack([np.full(N, x_min), np.linspace(y_min, y_max, N)]),  # Left boundary
        np.column_stack([np.full(N, x_max), np.linspace(y_min, y_max, N)])   # Right boundary
    ])
    return boundary_points

class DualPINN(torch.nn.Module):
    def __init__(self,
                 alignment_mode:str,
                 hidden_units:list,
                 mom_weight:float=1.,
                 div_weight:float=1.,
                 alignment_weight:float=1.,
                 bc_weight:float=1.,
                 div_activation:nn.Module=nn.Tanh(),
                 mom_activation:nn.Module=nn.Tanh(),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        self.in_dim = 2
        self.bc_weight = bc_weight
        self.mom_weight = mom_weight
        self.div_weight = div_weight
        self.alignment_mode = alignment_mode
        self.alignment_weight = alignment_weight
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        # Divergence free network
        div_out_dim = 2
        self.div_out_dim = div_out_dim
        self.div_hidden_units = [self.in_dim] + hidden_units
    
        div_net = nn.Sequential()
        for i in range(len(self.div_hidden_units)-1):
            div_net.add_module(f'div_lin{i}', nn.Linear(self.div_hidden_units[i], self.div_hidden_units[i+1]))
            div_net.add_module(f'div_act{i}', div_activation)
        div_net.add_module(f'div_lin{len(self.div_hidden_units)-1}', nn.Linear(self.div_hidden_units[-1], self.div_out_dim))
        
        self.div_net = div_net.to(self.device)
        
        # Momentum equation network # TODO: if it does not work well, split in two
        mom_out_dim = 3
        self.mom_out_dim = mom_out_dim
        self.mom_hidden_units = [self.in_dim] + hidden_units
        mom_net = nn.Sequential()
        for i in range(len(self.mom_hidden_units)-1):
            mom_net.add_module(f'mom_lin{i}', nn.Linear(self.mom_hidden_units[i], self.mom_hidden_units[i+1]))
            mom_net.add_module(f'mom_act{i}', mom_activation)
        mom_net.add_module(f'mom_lin{len(self.mom_hidden_units)-1}', nn.Linear(self.mom_hidden_units[-1], mom_out_dim))
        
        self.mom_net = mom_net.to(self.device)
                
        # Print the number of parameters
        div_params = sum(p.numel() for p in self.div_net.parameters())
        mom_params = sum(p.numel() for p in self.mom_net.parameters())
        total_params = div_params + mom_params
        print(f"Number of parameters in div_net: {div_params}")
        print(f"Number of parameters in mom_net: {mom_params}")
        print(f"Total number of parameters: {total_params}")
        
        self.device = device

    def forward(self, tx, return_final=False):
        # Get the predictions
        div_out = self.div_net(tx)
        mom_out = self.mom_net(tx)
        if return_final:
            return mom_out

        return torch.cat((mom_out.reshape(-1,3), div_out.reshape(-1,2)), dim=1).squeeze(0)


    def calc_alignment(
        self, x_pde:torch.Tensor, **kwargs
    ):
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        
        if self.alignment_mode == "DERL":
            return Dy_pred.reshape((x_pde.shape[0], -1))
        elif self.alignment_mode == "SOB":
            return torch.concat((Dy_pred.reshape((x_pde.shape[0], -1)), y_pred.reshape((x_pde.shape[0], -1))), dim=1)
        elif self.alignment_mode == "OUTL":
            return y_pred.reshape((x_pde.shape[0], -1))
        
    def calc_alignment_loss(
        self, x_pde:torch.Tensor, **kwargs
    ):
        y_pred = self.forward(x_pde)
        y_mom_pred = y_pred[:,:3]
        y_div_pred = y_pred[:,3:]

        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        Dy_mom_pred = Dy_pred[:,:3,:]
        Dy_div_pred = Dy_pred[:,3:,:]
        
        # Now we impose consistency between the branches
        if self.alignment_mode ==  'DERL':
            # Match the derivatives between the branches
            alignment_loss = self.alignment_loss_container(Dy_div_pred[:,:] - Dy_mom_pred[:,:2], torch.zeros_like(Dy_div_pred[:,:]))
        elif self.alignment_mode == 'OUTL':
            # Match the outputs between the branches
            alignment_loss = self.alignment_loss_container(y_div_pred - y_mom_pred[:,:2], torch.zeros_like(y_div_pred))
        elif self.alignment_mode == 'SOB':
            # It is the sum of the other two
            alignment_loss = self.alignment_loss_container(Dy_div_pred - Dy_mom_pred[:,:2,:], torch.zeros_like(Dy_div_pred))
            alignment_loss = alignment_loss + self.alignment_loss_container(y_div_pred - y_mom_pred[:,:2], torch.zeros_like(y_div_pred))
        
        return alignment_loss

    def calc_bc(self, x_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return torch.concat((y_bc_pred[:,:2], y_bc_pred[:,3:]), dim=1)

    def eval_bc_loss(self, x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        y_bc_mom = y_bc_pred[:,:3]
        return self.loss_container(y_bc_mom, y_bc)

    def calc_bc_loss(self, x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        y_bc_mom = y_bc_pred[:,:3]
        y_bc_div = y_bc_pred[:,3:]
        return self.loss_container(y_bc_mom, y_bc) + self.loss_container(y_bc_div, y_bc[:,:2])

    def calc_mom(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the prediction
        y_pred = self.forward(x_pde)[:,:3]
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward))(x_pde)[:,:3]
        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        lapl_u = torch.diagonal(Hy_pred[:,:2,:,:], dim1=2, dim2=3).sum(dim=2)
        mom_pde = - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,:], y_pred[:,:2]) + Dy_pred[:,-1,:]
        return mom_pde
    

    def calc_mom_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        mom_pde = self.calc_mom(x_pde)
        mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
        return mom_loss
    
    def eval_mom_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        mom_pde = self.calc_mom(x_pde)
        mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
        return mom_loss

    def calc_div(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,3:]
        # Calculate the divergence
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
        return div_pde
    
    def eval_div_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:2]
        # Calculate the divergence
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
        return self.loss_container(div_pde, torch.zeros_like(div_pde))

    def calc_div_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,3:]
        # Calculate the divergence
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
        div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
        return div_loss
    
    def loss_fn(self,
                x_pde:torch.Tensor,
                x_bc:torch.Tensor, y_bc: torch.Tensor, **kwargs
        ) -> torch.Tensor:


        bc_loss = self.calc_bc_loss(x_bc, y_bc)

        mom_loss = self.calc_mom_loss(x_pde)
        div_loss = self.calc_div_loss(x_pde)
        
        alignment_loss = self.calc_alignment_loss(x_pde)
    
        return bc_loss*self.bc_weight + self.alignment_weight*alignment_loss + self.mom_weight*mom_loss + self.div_weight*div_loss
    
    def eval_losses(self, 
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor,
        ) -> torch.Tensor:
        
        bc_loss = self.eval_bc_loss(x_bc, y_bc)
        mom_loss = self.eval_mom_loss(x_pde)
        div_loss = self.eval_div_loss(x_pde)
        alignment_loss = self.calc_alignment_loss(x_pde)
        y_pred = self.forward(x_pde)[:,:3]
        
        y_loss = self.loss_container(y_pred, y_pde)

        tot_loss_val = bc_loss + mom_loss + div_loss + alignment_loss
        return mom_loss, div_loss, y_loss, bc_loss, alignment_loss, tot_loss_val
    
    def evaluate_pde_residuals(self, x_pde:torch.Tensor):
        y_pred = self.forward(x_pde)[:,:3]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward))(x_pde)[:,:3]

        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        lapl_u = torch.diagonal(Hy_pred[:,:2,:,:], dim1=2, dim2=3).sum(dim=2)
        mom_pde = - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,:], y_pred[:,:2]) + Dy_pred[:,-1,:]
        
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,:])
        return torch.abs(mom_pde), torch.abs(div_pde)
    

