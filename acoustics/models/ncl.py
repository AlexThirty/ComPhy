import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian, jacfwd
from .params import rho, bulk, sound_speed, impedance

class NCL(torch.nn.Module):
    def __init__(self,
                 hidden_units: list=[32, 32, 32],
                 velx_weight:float=1.,
                 vely_weight:float=1.,
                 ic_weight:float=1.,
                 bc_weight:float=1.,
                 lr:float=1e-3,
                 activation:nn.Module=nn.Softplus(beta=20.),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        # Save the parameters
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.velx_weight = velx_weight
        self.vely_weight = vely_weight
        self.hidden_units = hidden_units
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        self.in_dim = 3
        self.out_dim = 3
        self.mat_dim = self.out_dim*(self.out_dim-1)//2
        
        self.hidden_list = [self.in_dim] + hidden_units
        pres_net = nn.Sequential()
        # Now the pres conservation network
        for i in range(len(self.hidden_list)-1):
            pres_net.add_module(f'pres_lin{i}', nn.Linear(in_features=self.hidden_list[i], out_features=self.hidden_list[i+1]))
            pres_net.add_module(f'pres_act{i}', activation)
        pres_net.add_module(f'pres_lin{len(self.hidden_list)-1}', nn.Linear(in_features=self.hidden_list[-1], out_features=self.mat_dim))
        # Save the network
        self.pres_net = pres_net.to(self.device)        
        # Save the optimizer
        self.lr = lr
        self.device = device
    
    def forward(self, x:torch.Tensor, return_final:bool=False) -> torch.Tensor:
        # Subfunction that outputs the matrix that parametrizes the divergence-free field
        def pres_A_matrix(x:torch.Tensor):
            # Pass through the networks
            pres_in = x
            pres_out = self.pres_net(pres_in)
            # Reshape into a matrix form
            mat = torch.zeros((self.out_dim, self.out_dim), device=self.device)
            triu_indexes = torch.triu_indices(self.out_dim, self.out_dim, offset=1)
            mat = mat.index_put(tuple(triu_indexes), pres_out)
            # Make the matrix antisymmetric
            A = mat - torch.transpose(mat, dim0=0, dim1=1)
            return A
        
        pres_fun = vmap(jacrev(pres_A_matrix))
        pres_out = torch.einsum('...ii', pres_fun(x.reshape((-1,3)))[:,:,:,:]).squeeze(0)
        return pres_out

    def calc_ic_loss(self,
                     x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return self.loss_container(y_ic_pred, y_ic)

    def calc_ic(self, x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return y_ic_pred

    def eval_ic_loss(self,
                     x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return self.loss_container(y_ic_pred, y_ic)
    
    def calc_bc_loss(self,
                     x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return self.loss_container(y_bc_pred, y_bc)

    def calc_bc(self, x_bc:torch.Tensor, y_bc:torch.Tensor=None, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return y_bc_pred

    def eval_bc_loss(self,
                     x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return self.loss_container(y_bc_pred, y_bc)

    def calc_velx_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        velx_pde = Dy_pred[:,1,0] + Dy_pred[:,0,1]/rho
        return self.loss_container(velx_pde, torch.zeros_like(velx_pde))

    def calc_velx(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        velx_pde = Dy_pred[:,1,0] + Dy_pred[:,0,1]/rho
        return velx_pde

    def eval_velx_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        velx_pde = self.calc_velx(x_pde)
        return self.loss_container(velx_pde, torch.zeros_like(velx_pde))

    def calc_vely_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        vely_pde = Dy_pred[:,2,0] + Dy_pred[:,0,2]/rho
        return self.loss_container(vely_pde, torch.zeros_like(vely_pde))

    def calc_vely(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        vely_pde = Dy_pred[:,2,0] + Dy_pred[:,0,2]/rho
        return vely_pde

    def eval_vely_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        vely_pde = self.calc_vely(x_pde)
        return self.loss_container(vely_pde, torch.zeros_like(vely_pde))

    def loss_fn(self,
                x_pde:torch.Tensor,
                x_ic:torch.Tensor, y_ic:torch.Tensor,
                x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        # Compute the PDE loss
        velx_pde_loss = self.calc_velx_loss(x_pde)
        vely_pde_loss = self.calc_vely_loss(x_pde)
        
        # Calculate boundary value
        bc_loss = self.calc_bc_loss(x_bc, y_bc)

        # Compute the initial loss
        ic_loss = self.calc_ic_loss(x_ic, y_ic)

        # Total final loss
        tot_loss = self.velx_weight*velx_pde_loss + self.vely_weight*vely_pde_loss + self.bc_weight*bc_loss + self.ic_weight*ic_loss
        
        return tot_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor):
        # Get the prediction

        bc_loss = self.eval_bc_loss(x_bc, y_bc)

        ic_loss = self.eval_ic_loss(x_ic, y_ic)
                
        # Now calculate the error wrt the true pdeution
        y_pred_final = self.forward(x_pde)
        y_loss = self.loss_container(y_pred_final, y_pde)

        

        velx_pde_loss = self.eval_velx_loss(x_pde)

        vely_pde_loss = self.eval_vely_loss(x_pde)

        tot_loss = bc_loss + ic_loss + velx_pde_loss + vely_pde_loss
        
        return torch.zeros_like(y_loss), velx_pde_loss, vely_pde_loss, y_loss, bc_loss, ic_loss, torch.zeros_like(y_loss), tot_loss
    
        
        
        
    def evaluate_consistency(self, x):
        # Get the derivatives 
        Dy_pred = vmap(jacrev(self.forward))(x)
        
        # Get the properties
        
        # Remember the equations
        # Sound speed
        # cc = np.sqrt(bulk_modulus/density)
        # Impedance
        # Z = density*cc
        # density = Z/cc
        # bulk_modulus = Z*cc
        # Acoustic equation
        # p_t + K (u_x + v_y) & = 0 \\ 
        #    u_t + p_x / \rho & = 0 \\
        #    v_t + p_y / \rho & = 0.
        
        # pres PDE
        pres_pde = Dy_pred[:,0,0] + bulk*(Dy_pred[:,1,1] + Dy_pred[:,2,2])
        
        # Velocity PDE
        velx_pde = Dy_pred[:,1,0] + Dy_pred[:,0,1]/rho
        vely_pde = Dy_pred[:,2,0] + Dy_pred[:,0,2]/rho
        
        # Compute the PDE loss
        pres_pde_loss = self.loss_container(pres_pde, torch.zeros_like(pres_pde))
        velx_pde_loss = self.loss_container(velx_pde, torch.zeros_like(velx_pde))
        vely_pde_loss = self.loss_container(vely_pde, torch.zeros_like(vely_pde))
        
        pde_loss = pres_pde_loss + velx_pde_loss + vely_pde_loss
        
        return torch.abs(pres_pde), torch.abs(velx_pde), torch.abs(vely_pde)    
        
        
