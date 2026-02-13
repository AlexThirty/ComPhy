import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian, jacfwd
from .params import rho, bulk, sound_speed, impedance

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

def generate_random_points(num_points: int, xmin: float, xmax: float, ymin:float, ymax:float, tmin:float, tmax:float) -> torch.Tensor:
    x = torch.rand(num_points, 3)
    x[:, 0] = x[:, 0] * (tmax - tmin) + tmin # Scale time component to [0, t]
    # Scale x component to [xmin, xmax]
    x[:, 1] = x[:, 1] * (xmax - xmin) + xmin
    x[:, 2] = x[:, 2] * (ymax - ymin) + ymin
    return x

gamma = 1.4

class TriplePINN(torch.nn.Module):
    def __init__(self,
                 alignment_mode,
                 hidden_units: list=[32, 32, 32],
                 alignment_weight:float=1.,
                 pres_weight:float=1.,
                 velx_weight:float=1.,
                 vely_weight:float=1.,
                 ic_weight:float=1.,
                 bc_weight:float=1.,
                 lr:float=1e-3,
                 activation:nn.Module=nn.Tanh(),
                 device: str='cuda:0',
                 special_bc_loss:bool=False,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        # Save the parameters
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.alignment_weight = alignment_weight
        self.pres_weight = pres_weight
        self.velx_weight = velx_weight
        self.vely_weight = vely_weight
        self.hidden_units = hidden_units
        self.alignment_mode = alignment_mode
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        self.in_dim = 3
        self.out_dim = 3
        self.mat_dim = self.out_dim*(self.out_dim-1)//2
        
        self.hidden_list = [self.in_dim] + hidden_units
        pressure_net = nn.Sequential()
        # Now the pressure conservation network
        for i in range(len(self.hidden_list)-1):
            pressure_net.add_module(f'pressure_lin{i}', nn.Linear(in_features=self.hidden_list[i], out_features=self.hidden_list[i+1]))
            pressure_net.add_module(f'pressure_act{i}', activation)
        pressure_net.add_module(f'pressure_lin{len(self.hidden_list)-1}', nn.Linear(in_features=self.hidden_list[-1], out_features=self.out_dim))
        # Save the network
        self.pressure_net = pressure_net.to(self.device)
        
        velx_net = nn.Sequential()
        # Now the momentum conservation network
        for i in range(len(self.hidden_list)-1):
            velx_net.add_module(f'velx_lin{i}', nn.Linear(in_features=self.hidden_list[i], out_features=self.hidden_list[i+1]))
            velx_net.add_module(f'velx_act{i}', activation)
        velx_net.add_module(f'velx_lin{len(self.hidden_list)-1}', nn.Linear(in_features=self.hidden_list[-1], out_features=self.out_dim-1))
        # Save the network
        self.velx_net = velx_net.to(self.device)
        
        vely_net = nn.Sequential()
        # Now the momentum conservation network
        for i in range(len(self.hidden_list)-1):
            vely_net.add_module(f'vely_lin{i}', nn.Linear(in_features=self.hidden_list[i], out_features=self.hidden_list[i+1]))
            vely_net.add_module(f'vely_act{i}', activation)
        vely_net.add_module(f'vely_lin{len(self.hidden_list)-1}', nn.Linear(in_features=self.hidden_list[-1], out_features=self.out_dim-1))
        # Save the network
        self.vely_net = vely_net.to(self.device)
        
        self.special_bc_loss = special_bc_loss
        
        # Save the optimizer
        self.lr = lr
        
        self.device = device
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, x:torch.Tensor, return_final:bool=False) -> torch.Tensor:
        pressure_out = self.pressure_net(x)
        velx_out = self.velx_net(x)
        vely_out = self.vely_net(x)
        
        if return_final:
            return pressure_out

        return torch.cat((pressure_out.reshape((-1,3)), velx_out.reshape((-1,2)), vely_out.reshape((-1,2))), dim=1).squeeze(0)
    
    
    def calc_alignment_loss(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        
        # Get the predictions
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)

        pressure_pred = y_pred[:,:3]
        velx_pred = y_pred[:,3:5]
        vely_pred = y_pred[:,5:]

        pressure_Dy_pred = Dy_pred[:,:3]
        velx_Dy_pred = Dy_pred[:,3:5]
        vely_Dy_pred = Dy_pred[:,5:]
        

        if self.alignment_mode == 'DERL':
            out1 = pressure_Dy_pred[:,:2] - torch.column_stack((velx_Dy_pred[:,1:], velx_Dy_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_Dy_pred[:,2:3], pressure_Dy_pred[:,0:1]/rho)) - vely_Dy_pred[:,:2]
            alignment_loss = self.loss_container(
                out1, torch.zeros_like(out1)
            ) + self.loss_container(
                out2, torch.zeros_like(out2)
            ) 
            
        elif self.alignment_mode == 'OUTL':
            out1 = pressure_pred[:,:2] - torch.column_stack((velx_pred[:,1:], velx_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_pred[:,2:3], pressure_pred[:,0:1]/rho)) - vely_pred[:,:2]
            alignment_loss = self.loss_container(
                out1, torch.zeros_like(out1)
            ) + self.loss_container(
                out2, torch.zeros_like(out2)
            )
                
        elif self.alignment_mode == 'SOB':
            out1 = pressure_Dy_pred[:,:2] - torch.column_stack((velx_Dy_pred[:,1:], velx_Dy_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_Dy_pred[:,2:3], pressure_Dy_pred[:,0:1]/rho)) - vely_Dy_pred[:,:2]
            alignment_loss = self.loss_container(
                out1, torch.zeros_like(out1)
            ) + self.loss_container(
                out2, torch.zeros_like(out2)
            ) 
            
            out1 = pressure_pred[:,:2] - torch.column_stack((velx_pred[:,1:], velx_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_pred[:,2:3], pressure_pred[:,0:1]/rho)) - vely_pred[:,:2]
            alignment_loss += self.loss_container(
                out1, torch.zeros_like(out1)
            ) + self.loss_container(
                out2, torch.zeros_like(out2)
            )
            
        else:
            raise ValueError('Invalid alignment mode')
    
        return alignment_loss
    
    def calc_alignment(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        # Get the predictions
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)

        pressure_pred = y_pred[:,:3]
        velx_pred = y_pred[:,3:5]
        vely_pred = y_pred[:,5:]

        pressure_Dy_pred = Dy_pred[:,:3]
        velx_Dy_pred = Dy_pred[:,3:5]
        vely_Dy_pred = Dy_pred[:,5:]
        

        if self.alignment_mode == 'DERL':
            out1 = pressure_Dy_pred[:,:2] - torch.column_stack((velx_Dy_pred[:,1:], velx_Dy_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_Dy_pred[:,2:3], pressure_Dy_pred[:,0:1]/rho)) - vely_Dy_pred[:,:2]
            prov = torch.column_stack((out1.reshape((x_pde.shape[0], -1)), out2.reshape((x_pde.shape[0], -1))))
            return prov

        elif self.alignment_mode == 'OUTL':
            out1 = pressure_pred[:,:2] - torch.column_stack((velx_pred[:,1:], velx_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_pred[:,2:3], pressure_pred[:,0:1]/rho)) - vely_pred[:,:2]
            return torch.column_stack((out1.reshape((x_pde.shape[0], -1)), out2.reshape((x_pde.shape[0], -1))))
        
        elif self.alignment_mode == 'SOB':
            out1 = pressure_Dy_pred[:,:2] - torch.column_stack((velx_Dy_pred[:,1:], velx_Dy_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_Dy_pred[:,2:3], pressure_Dy_pred[:,0:1]/rho)) - vely_Dy_pred[:,:2]
            prov = torch.column_stack((out1.reshape((x_pde.shape[0], -1)), out2.reshape((x_pde.shape[0], -1))))
            
            out1 = pressure_pred[:,:2] - torch.column_stack((velx_pred[:,1:], velx_pred[:,0:1]/rho))
            out2 = torch.column_stack((pressure_pred[:,2:3], pressure_pred[:,0:1]/rho)) - vely_pred[:,:2]
            return torch.column_stack((prov, out1.reshape((x_pde.shape[0], -1)), out2.reshape((x_pde.shape[0], -1))))
            
        else:
            raise ValueError('Invalid alignment mode')
        
        
    def calc_ic_loss(self,
                     x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        pressure_ic_pred = y_ic_pred[:,:3]
        velx_ic_pred = y_ic_pred[:,3:5]
        vely_ic_pred = y_ic_pred[:,5:]

        # Compute the initial loss
        ic_loss = self.loss_container(
            pressure_ic_pred, torch.column_stack((y_ic[:,0], y_ic[:,1], y_ic[:,2]))
        )
        ic_loss += self.loss_container(
            velx_ic_pred, torch.column_stack((y_ic[:,1], y_ic[:,0]))
        )
        ic_loss += self.loss_container(
            vely_ic_pred, torch.column_stack((y_ic[:,2], y_ic[:,0]))
        )
        
        return ic_loss

    def calc_ic(self, x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return y_ic_pred

    def eval_ic_loss(self,
                     x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)[:,:3]
        return self.loss_container(y_ic_pred, y_ic)
    
    def calc_bc_loss(self,
                     x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        pressure_bc_pred = y_bc_pred[:,:3]
        velx_bc_pred = y_bc_pred[:,3:5]
        vely_bc_pred = y_bc_pred[:,5:]
        # Compute the boundary loss
        bc_loss = self.loss_container(
            pressure_bc_pred, torch.column_stack((y_bc[:,0], y_bc[:,1], y_bc[:,2]))
        )
        bc_loss += self.loss_container(
            velx_bc_pred, torch.column_stack((y_bc[:,1], y_bc[:,0]))
        )
        bc_loss += self.loss_container(
            vely_bc_pred, torch.column_stack((y_bc[:,2], y_bc[:,0]))
        )
        
        return bc_loss

    def calc_bc(self, x_bc:torch.Tensor, y_bc:torch.Tensor=None, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)
        return y_bc_pred

    def eval_bc_loss(self,
                     x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs) -> torch.Tensor:
        y_bc_pred = self.forward(x_bc)[:,:3]
        return self.loss_container(y_bc_pred, y_bc)
        
    def calc_pres(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        pressure_pde = Dy_pred[:,0,0] + bulk*(Dy_pred[:,1,1] + Dy_pred[:,2,2])
        return pressure_pde

    def calc_pres_loss(self,
        x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        pressure_pde = self.calc_pres(x_pde)
        return self.loss_container(pressure_pde, torch.zeros_like(pressure_pde))
    
    def eval_pres_loss(self,
        x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        pressure_pde = self.calc_pres(x_pde)
        return self.loss_container(pressure_pde, torch.zeros_like(pressure_pde))

    def calc_velx_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,3:5]
        velx_pde = Dy_pred[:,0,0] + Dy_pred[:,1,1]/rho
        return self.loss_container(velx_pde, torch.zeros_like(velx_pde))

    def calc_velx(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,3:5]
        velx_pde = Dy_pred[:,0,0] + Dy_pred[:,1,1]/rho
        return velx_pde

    def eval_velx_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        velx_pde = Dy_pred[:,1,0] + Dy_pred[:,0,1]/rho
        return self.loss_container(velx_pde, torch.zeros_like(velx_pde))

    def calc_vely_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,5:]
        vely_pde = Dy_pred[:,0,0] + Dy_pred[:,1,2]/rho
        return self.loss_container(vely_pde, torch.zeros_like(vely_pde))

    def calc_vely(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,5:]
        vely_pde = Dy_pred[:,0,0] + Dy_pred[:,1,2]/rho
        return vely_pde

    def eval_vely_loss(self, x_pde: torch.Tensor, **kwargs) -> torch.Tensor:
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]
        vely_pde = Dy_pred[:,2,0] + Dy_pred[:,0,2]/rho
        return self.loss_container(vely_pde, torch.zeros_like(vely_pde))

    def loss_fn(self,
                x_pde:torch.Tensor,
                x_ic:torch.Tensor, y_ic:torch.Tensor,
                x_bc:torch.Tensor, y_bc:torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        # Compute the PDE loss
        pressure_pde_loss = self.calc_pres_loss(x_pde)
        velx_pde_loss = self.calc_velx_loss(x_pde)
        vely_pde_loss = self.calc_vely_loss(x_pde)
        
        # Calculate boundary value
        bc_loss = self.calc_bc_loss(x_bc, y_bc)

        # Compute the icial loss
        ic_loss = self.calc_ic_loss(x_ic, y_ic)
        
        # Compute the alignment loss
        alignment_loss = self.calc_alignment_loss(x_pde)

        # Total final loss
        tot_loss = self.pres_weight*pressure_pde_loss + self.velx_weight*velx_pde_loss + self.vely_weight*vely_pde_loss + self.bc_weight*bc_loss + self.ic_weight*ic_loss + self.alignment_weight*alignment_loss

        return tot_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor,
                    x_bc:torch.Tensor, y_bc:torch.Tensor,
                    **kwargs) -> tuple:
        # Get the prediction

        bc_loss = self.eval_bc_loss(x_bc, y_bc)

        ic_loss = self.eval_ic_loss(x_ic, y_ic)
                
        # Now calculate the error wrt the true solution
        y_pred_final = self.forward(x_pde)[:,:3]
        y_loss = self.loss_container(y_pred_final, y_pde)

        
        pressure_pde_loss = self.eval_pres_loss(x_pde)

        velx_pde_loss = self.eval_velx_loss(x_pde)

        vely_pde_loss = self.eval_vely_loss(x_pde)
        
        alignment_loss = self.calc_alignment_loss(x_pde)

        tot_loss = bc_loss + ic_loss + pressure_pde_loss + velx_pde_loss + vely_pde_loss + alignment_loss

        return pressure_pde_loss, velx_pde_loss, vely_pde_loss, y_loss, bc_loss, ic_loss, alignment_loss, tot_loss
        
        
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
        
        # Pressure PDE
        pressure_pde = Dy_pred[:,0,0] + bulk*(Dy_pred[:,1,1] + Dy_pred[:,2,2])
        
        # Velocity PDE
        velx_pde = Dy_pred[:,1,0] + Dy_pred[:,0,1]/rho
        vely_pde = Dy_pred[:,2,0] + Dy_pred[:,0,2]/rho
        
        # Compute the PDE loss
        pressure_pde_loss = self.loss_container(pressure_pde, torch.zeros_like(pressure_pde))
        velx_pde_loss = self.loss_container(velx_pde, torch.zeros_like(velx_pde))
        vely_pde_loss = self.loss_container(vely_pde, torch.zeros_like(vely_pde))
        
        pde_loss = pressure_pde_loss + velx_pde_loss + vely_pde_loss
        
        return torch.abs(pressure_pde), torch.abs(velx_pde), torch.abs(vely_pde)   