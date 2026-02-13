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
nu = 0.1  # Viscosity

class NCL(torch.nn.Module):
    def __init__(self,
                 hidden_units:list,
                 mom_weight:float=1.,
                 init_weight:float=1.,
                 bc_weight:float=1.,
                 ic_weight:float=1.,
                 lr:float=1e-3,
                 div_activation:nn.Module=nn.Softplus(beta=20.),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        self.in_dim = 3
        self.bc_weight = bc_weight
        self.mom_weight = mom_weight
        self.init_weight = init_weight
        self.ic_weight = ic_weight
        self.div_hidden_units = hidden_units
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        # Divergence free network
        div_out_dim = 2
        self.div_out_dim = div_out_dim
        self.div_mat_dim = (div_out_dim*(div_out_dim-1))//2
        self.div_hidden_units = [self.in_dim] + hidden_units

        div_net = nn.Sequential()
        for i in range(len(self.div_hidden_units)-1):
            div_net.add_module(f'div_lin{i}', nn.Linear(self.div_hidden_units[i], self.div_hidden_units[i+1]))
            div_net.add_module(f'div_act{i}', div_activation)
        div_net.add_module(f'div_lin{len(self.div_hidden_units)-1}', nn.Linear(self.div_hidden_units[-1], self.div_mat_dim+1))
        
        self.div_net = div_net.to(self.device)
        
        # Save the optimizer
        self.lr = lr
        
        self.device = device
    
    def forward(self, tx, return_final=False):
        def div_A_matrix(x:torch.Tensor):
            #print(x.shape)
            # Pass through the networks
            #root_out = self.root_net(x)
            div_in = x
            div_out = self.div_net(div_in)[:-1]
            # Reshape into a matrix form
            mat = torch.zeros((self.div_out_dim, self.div_out_dim), device=self.device)
            triu_indexes = torch.triu_indices(self.div_out_dim, self.div_out_dim, offset=1)
            mat = mat.index_put(tuple(triu_indexes), div_out)
            #print(out.shape)
            # Make the matrix antisymmetric
            A = mat - torch.transpose(mat, dim0=0, dim1=1)
            #print(A.shape)
            return A
        # Now get the vector
        # div_vec has a (b,3,3,4) shape
        div_fun = vmap(jacrev(div_A_matrix))
        # This is the divergence free output of the divergence equation
        div_out = torch.einsum('...ii', div_fun(tx.reshape((-1,3)))[:,:,:,1:])
        p_div = self.div_net(tx)
        if p_div.ndim > 1:
            p_div = p_div[:,-1:]
        else:
            p_div = p_div[-1]

        return torch.concat((div_out.reshape((-1,2)), p_div.reshape((-1,1))), dim=1).squeeze(0)

    def calc_bc(self, x_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return y_bc_pred

    def eval_bc_loss(self, x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        y_bc_mom = y_bc_pred[:,:3]
        return self.loss_container(y_bc_mom, y_bc)

    def calc_bc_loss(self, x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return self.loss_container(y_bc_pred, y_bc)

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
    
    def loss_fn(self,
                x_pde:torch.Tensor,
                x_bc:torch.Tensor, y_bc: torch.Tensor,
                x_ic:torch.Tensor, y_ic: torch.Tensor, **kwargs
        ) -> torch.Tensor:


        bc_loss = self.calc_bc_loss(x_bc, y_bc)
        ic_loss = self.calc_ic_loss(x_ic, y_ic)
        mom_loss = self.calc_mom_loss(x_pde)
        return bc_loss*self.bc_weight + self.mom_weight*mom_loss + self.ic_weight*ic_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor
        ) -> torch.Tensor:
        
        bc_loss = self.eval_bc_loss(x_bc, y_bc)
        ic_loss = self.eval_ic_loss(x_ic, y_ic)
        mom_loss = self.eval_mom_loss(x_pde)
        y_pred = self.forward(x_pde)[:,:3]
        
        y_loss = self.loss_container(y_pred, y_pde)

        tot_loss_val = bc_loss + mom_loss + ic_loss
        return mom_loss, torch.zeros_like(mom_loss), y_loss, bc_loss, ic_loss, torch.zeros_like(mom_loss), tot_loss_val
    
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
    
