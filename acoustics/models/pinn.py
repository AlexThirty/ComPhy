import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian, jacfwd
from .params import bulk, rho, sound_speed, impedance

def generate_boundary_points(num_points: int, t: float) -> torch.Tensor:
    x = torch.rand(num_points, 2)
    x[:, 0] = x[:, 0] * t  # Scale time component to [0, t]
    x[:, 1] = torch.randint(0, 2, (num_points,)).float() * 2 - 1  # Boundary points at -1 or 1
    return x

def generate_random_points(num_points: int, xmin: float, xmax: float, ymin:float, ymax:float, tmin:float, tmax:float) -> torch.Tensor:
    x = torch.rand(num_points, 3)
    x[:, 0] = x[:, 0] * (tmax - tmin) + tmin # Scale time component to [0, t]
    # Scale x component to [xmin, xmax]
    x[:, 1] = x[:, 1] * (xmax - xmin) + xmin
    x[:, 2] = x[:, 2] * (ymax - ymin) + ymin
    return x

gamma = 1.4

class PINN(torch.nn.Module):
    def __init__(self,
                 hidden_units: list=[32, 32, 32],
                 pres_weight:float=1.,
                 velx_weight:float=1.,
                 vely_weight:float=1.,
                 ic_weight:float=1.,
                 bc_weight:float=1.,
                 lr:float=1e-3,
                 activation:nn.Module=nn.Tanh(),
                 device: str='cuda:0',
                 special_bc_loss:bool=False,
                 tmax = 0.9,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        self.tmax = tmax
        # Save the parameters
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.pres_weight = pres_weight
        self.velx_weight = velx_weight
        self.vely_weight = vely_weight
        self.hidden_units = hidden_units
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        
        self.in_dim = 3
        self.out_dim = 3
        

        # Define the pres conservation network
        net_dict = OrderedDict(
            {'pres_lin0': nn.Linear(self.in_dim, hidden_units[0]),
            'pres_act0': activation}
        )
        # Define the net, hidden layers
        for i in range(1, len(self.hidden_units)):
            net_dict.update({f'pres_lin{i}': nn.Linear(in_features=self.hidden_units[i-1], out_features=self.hidden_units[i])})
            net_dict.update({f'pres_act{i}': activation})
        # Add the last layer
        net_dict.update({f'pres_lin{len(self.hidden_units)}': nn.Linear(in_features=self.hidden_units[-1], out_features=self.out_dim)})
        # Save the network
        self.net = nn.Sequential(net_dict).to(self.device)
        
        self.special_bc_loss = special_bc_loss
        
        # Save the optimizer
        self.lr = lr
        
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        #self.opt = torch.optim.LBFGS(self.parameters(), lr=1)
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt, milestones=[25000, 50000, 75000, 100000, 150000], gamma=5e-1)
        # Device
        self.device = device
                
                
    def forward(self, x:torch.Tensor, return_final:bool=False) -> torch.Tensor:
        # In the PINN case the output is standard
        return self.net(x)
        
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

    def calc_pres_loss(self,
        x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        
    def calc_pres(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        pres_pde = Dy_pred[:,0,0] + bulk*(Dy_pred[:,1,1] + Dy_pred[:,2,2])
        return pres_pde

    def calc_pres_loss(self,
        x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        pres_pde = self.calc_pres(x_pde)
        return self.loss_container(pres_pde, torch.zeros_like(pres_pde))
    
    def eval_pres_loss(self,
        x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        pres_pde = self.calc_pres(x_pde)
        return self.loss_container(pres_pde, torch.zeros_like(pres_pde))

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
        pres_pde_loss = self.calc_pres_loss(x_pde)
        velx_pde_loss = self.calc_velx_loss(x_pde)
        vely_pde_loss = self.calc_vely_loss(x_pde)
        
        # Calculate boundary value
        bc_loss = self.calc_bc_loss(x_bc, y_bc)

        # Compute the initial loss
        ic_loss = self.calc_ic_loss(x_ic, y_ic)

        # Total final loss
        tot_loss = self.pres_weight*pres_pde_loss + self.velx_weight*velx_pde_loss + self.vely_weight*vely_pde_loss + self.bc_weight*bc_loss + self.ic_weight*ic_loss
        
        return tot_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs):
        # Get the prediction

        bc_loss = self.eval_bc_loss(x_bc, y_bc)

        ic_loss = self.eval_ic_loss(x_ic, y_ic)
                
        # Now calculate the error wrt the true pdeution
        y_pred_final = self.forward(x_pde)
        y_loss = self.loss_container(y_pred_final, y_pde)

        
        pres_pde_loss = self.eval_pres_loss(x_pde)

        velx_pde_loss = self.eval_velx_loss(x_pde)

        vely_pde_loss = self.eval_vely_loss(x_pde)

        tot_loss = bc_loss + ic_loss + pres_pde_loss + velx_pde_loss + vely_pde_loss
        
        return pres_pde_loss, velx_pde_loss, vely_pde_loss, y_loss, bc_loss, ic_loss, torch.zeros_like(y_loss), tot_loss
    
        
        
    def evaluate_consistency(self, x):
        # Get the predictions
        y_pred = self.forward(x)
        
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
        
        
