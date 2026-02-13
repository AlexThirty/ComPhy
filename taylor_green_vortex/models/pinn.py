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
nu = 0.1

class PINN(torch.nn.Module):
    def __init__(self,
                 hidden_units:list,
                 mom_weight:float=1.,
                 div_weight:float=1.,
                 bc_weight:float=1.,
                 ic_weight:float=1.,
                 lr:float=1e-3,
                 activation:nn.Module=nn.Tanh(),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        self.in_dim = 3
        self.bc_weight = bc_weight
        self.div_weight = div_weight
        self.mom_weight = mom_weight
        self.hidden_units = hidden_units
        self.ic_weight = ic_weight

        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        # Divergence free network
        out_dim = 3
        self.out_dim = out_dim
        self.hidden_units = [self.in_dim] + hidden_units
    
        net = nn.Sequential()
        for i in range(len(self.hidden_units)-1):
            net.add_module(f'div_lin{i}', nn.Linear(self.hidden_units[i], self.hidden_units[i+1]))
            net.add_module(f'div_act{i}', activation)
        net.add_module(f'div_lin{len(self.hidden_units)-1}', nn.Linear(self.hidden_units[-1], self.out_dim))
        
        self.net = net.to(self.device)

        # Save the optimizer
        self.lr = lr
        
        self.device = device
    
    def forward(self, tx, return_final=False):
        return self.net(tx)

    def calc_bc(self, x_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return y_bc_pred

    def eval_bc_loss(self, x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        y_bc_mom = y_bc_pred[:,:3]
        return self.loss_container(y_bc_mom, y_bc)

    def calc_bc_loss(self, x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        y_bc_mom = y_bc_pred[:,:3]
        return self.loss_container(y_bc_mom, y_bc)

    def calc_ic(self, x_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return y_ic_pred

    def eval_ic_loss(self, x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        y_ic_mom = y_ic_pred[:,:3]
        return self.loss_container(y_ic_mom, y_ic)

    def calc_ic_loss(self, x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        y_ic_mom = y_ic_pred[:,:3]
        return self.loss_container(y_ic_mom, y_ic)

    def calc_mom(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the prediction
        y_pred = self.forward(x_pde)[:,:3]
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward))(x_pde)[:,:3]
        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        lapl_u = torch.diagonal(Hy_pred[:,:2,1:,1:], dim1=2, dim2=3).sum(dim=2)
        mom_pde = Dy_pred[:,:2,0] - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,1:], y_pred[:,:2]) + Dy_pred[:,-1,1:]
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
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Calculate the divergence
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,1:])
        return div_pde
    
    def eval_div_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Calculate the divergence
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,1:])
        return self.loss_container(div_pde, torch.zeros_like(div_pde))

    def calc_div_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Calculate the divergence
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,1:])
        div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
        return div_loss
    
    def loss_fn(self,
                x_pde:torch.Tensor,
                x_bc:torch.Tensor, y_bc: torch.Tensor,
                x_ic:torch.Tensor, y_ic: torch.Tensor, **kwargs
        ) -> torch.Tensor:


        bc_loss = self.calc_bc_loss(x_bc, y_bc)
        ic_loss = self.calc_ic_loss(x_ic, y_ic)
        mom_loss = self.calc_mom_loss(x_pde)
        div_loss = self.calc_div_loss(x_pde)
        

        return bc_loss*self.bc_weight + self.mom_weight*mom_loss + self.div_weight*div_loss + self.ic_weight*ic_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor
        ) -> torch.Tensor:
        
        bc_loss = self.eval_bc_loss(x_bc, y_bc)
        ic_loss = self.eval_ic_loss(x_ic, y_ic)
        mom_loss = self.eval_mom_loss(x_pde)
        div_loss = self.eval_div_loss(x_pde)
        y_pred = self.forward(x_pde)[:,:3]
        
        y_loss = self.loss_container(y_pred, y_pde)

        tot_loss_val = bc_loss + mom_loss + div_loss + ic_loss
        return mom_loss, div_loss, y_loss, bc_loss, ic_loss, torch.zeros_like(mom_loss), tot_loss_val
    
    def evaluate_pde_residuals(self, x_pde:torch.Tensor):
        y_pred = self.forward(x_pde)[:,:3]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        # Get the hessians
        Hy_pred = vmap(hessian(self.forward))(x_pde)[:,:3]

        # Now we impose the pinn loss on the momentum branch
        # Calculate the pde_residual
        lapl_u = torch.diagonal(Hy_pred[:,:2,1:,1:], dim1=2, dim2=3).sum(dim=2)
        mom_pde = Dy_pred[:,:2,0] - nu*lapl_u + torch.einsum('bij,bj->bi', Dy_pred[:,:2,1:], y_pred[:,:2]) + Dy_pred[:,-1,1:]
        
        div_pde = torch.einsum('bii->b', Dy_pred[:,:2,1:])
        return torch.abs(mom_pde), torch.abs(div_pde)
    
