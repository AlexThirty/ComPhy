import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.func import vmap, jacrev, hessian, jacfwd

gamma = 5./3.

class SoftAbs(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(SoftAbs, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.abs(x)>1, x-0.5, 0.5*x**2)

class NCL(torch.nn.Module):
    def __init__(self,
                 hidden_units: list=[32, 32, 32],
                 cont_weight:float=1.,
                 mom_weight:float=1.,
                 state_weight:float=1.,
                 ind_weight:float=1.,
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
        self.cont_weight = cont_weight
        self.mom_weight = mom_weight
        self.state_weight = state_weight
        self.ind_weight = ind_weight
        self.hidden_units = hidden_units
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        self.in_dim = 5
        self.out_dim = 2
        self.mat_dim = self.out_dim*(self.out_dim-1)//2
        
        self.hidden_list = [self.in_dim] + hidden_units
        self.positivizer = SoftAbs()
        gauss_net = nn.Sequential()
        # Now the gauss conservation network
        for i in range(len(self.hidden_list)-1):
            gauss_net.add_module(f'gauss_lin{i}', nn.Linear(in_features=self.hidden_list[i], out_features=self.hidden_list[i+1]))
            gauss_net.add_module(f'gauss_act{i}', activation)
        gauss_net.add_module(f'gauss_lin{len(self.hidden_list)-1}', nn.Linear(in_features=self.hidden_list[-1], out_features=self.mat_dim+4))
        # Save the network
        self.gauss_net = gauss_net.to(self.device)        
        # Save the optimizer
        self.lr = lr
        self.device = device
        
    def embed(self, x:torch.Tensor):
        c = 2*np.pi
        x = x.reshape((-1,3))
        return torch.cat([x[:,:1], torch.cos(c*x[:,1:2]), torch.sin(c*x[:,1:2]), torch.cos(c*x[:,2:3]), torch.sin(c*x[:,2:3])], dim=1).squeeze(0)
    
    
    def forward(self, x:torch.Tensor, return_final:bool=False) -> torch.Tensor:
        # Subfunction that outputs the matrix that parametrizes the divergence-free field
        def gauss_A_matrix(x:torch.Tensor):
            # Pass through the networks
            gauss_in = self.embed(x)
            gauss_out = self.gauss_net(gauss_in)[-1:]
            # Reshape into a matrix form
            mat = torch.zeros((self.out_dim, self.out_dim), device=self.device)
            triu_indexes = torch.triu_indices(self.out_dim, self.out_dim, offset=1)
            mat = mat.index_put(tuple(triu_indexes), gauss_out)
            # Make the matrix antisymmetric
            A = mat - torch.transpose(mat, dim0=0, dim1=1)
            return A
        
        gauss_fun = vmap(jacrev(gauss_A_matrix))
        gauss_out = torch.einsum('...ii', gauss_fun(x.reshape((-1,3)))[:,:,:,1:]).squeeze(0)
        
        non_div_out = self.gauss_net(self.embed(x.reshape((-1,3))))
        if non_div_out.ndim == 1:
            non_div_out = non_div_out.unsqueeze(0)
        non_div_out = non_div_out[:, :-1].reshape((-1,4))

        #return torch.cat([self.positivizer(non_div_out[:,:1]), non_div_out[:,1:3], self.positivizer(non_div_out[:,3:4]), non_div_out[:,4:6], gauss_out.reshape((-1,2))], dim=1).squeeze(0)
        return torch.cat([non_div_out, gauss_out.reshape((-1,2))], dim=1).squeeze(0)
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
        
    def calc_cont(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        
        rho = y_pred[:,0]
        u = y_pred[:,1:3]
        P = y_pred[:,3]
        B = y_pred[:,4:6]

        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]
        dP = Dy_pred[:,3]
        DB = Dy_pred[:,4:6]

        cont_pde = Drho[:,0] + rho*(Du[:,0,1] + Du[:,1,2]) + Drho[:,1]*u[:,0] + Drho[:,2]*u[:,1]
        return cont_pde

    def calc_cont_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        cont_pde = self.calc_cont(x_pde)
        return self.loss_container(cont_pde, torch.zeros_like(cont_pde))

    def eval_cont_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        cont_pde = self.calc_cont(x_pde)
        return self.loss_container(cont_pde, torch.zeros_like(cont_pde))

    def calc_mom(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        
        rho = y_pred[:,0]
        u = y_pred[:,1:3]
        P = y_pred[:,3]
        B = y_pred[:,4:6]

        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]
        DP = Dy_pred[:,3]
        DB = Dy_pred[:,4:6]
        
        
        mom_pde_x = rho*(Du[:,0,0] + Du[:,0,1]*u[:,0] + Du[:,0,2]*u[:,1])
        mom_pde_x += (DP[:,1] - 2*B[:,0]*DB[:,0,1] - B[:,1]*DB[:,0,2] - B[:,0]*DB[:,1,2])
        #mom_pde_x = (Du[:,0,0] + Du[:,0,1]*u[:,0] + Du[:,0,2]*u[:,1])
        #mom_pde_x += (DP[:,1]/rho - 2*B[:,0]/rho*DB[:,0,1] - B[:,1]/rho*DB[:,0,2] - B[:,0]/rho*DB[:,1,2])
        
        mom_pde_y = rho*(Du[:,1,0] + Du[:,1,1]*u[:,0] + Du[:,1,2]*u[:,1])
        mom_pde_y += (DP[:,2] - 2*B[:,1]*DB[:,1,2] - B[:,1]*DB[:,0,1] - B[:,0]*DB[:,1,1])
        #mom_pde_y = (Du[:,1,0] + Du[:,1,1]*u[:,0] + Du[:,1,2]*u[:,1])
        #mom_pde_y += (DP[:,2]/rho - 2*B[:,1]/rho*DB[:,1,2] - B[:,0]/rho*DB[:,1,1] - B[:,1]/rho*DB[:,0,1])


        mom_pde = torch.stack([mom_pde_x, mom_pde_y], dim=1)
        return mom_pde

    def calc_mom_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        mom_pde = self.calc_mom(x_pde)
        return self.loss_container(mom_pde, torch.zeros_like(mom_pde))

    def eval_mom_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        mom_pde = self.calc_mom(x_pde)
        return self.loss_container(mom_pde, torch.zeros_like(mom_pde))

    def calc_state(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        
        
        rho = y_pred[:,0]
        u = y_pred[:,1:3]
        P = y_pred[:,3]
        B = y_pred[:,4:6]

        P_gas = P - 0.5 * (B[:,0]**2 + B[:,1]**2)

        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]
        DP = Dy_pred[:,3]
        DB = Dy_pred[:,4:6]


        state = DP[:,0] + (gamma*P_gas + B[:,1]**2)*Du[:,0,1]
        state -= B[:,0]*B[:,1]*(Du[:,1,1])
        state += DP[:,1]*u[:,0]
        state += (gamma-2)*(u[:,0]*B[:,0] + u[:,1]*B[:,1])*(DB[:,0,1])
        state -= B[:,0]*B[:,1]*(Du[:,0,2])
        state += (gamma*P_gas + B[:,0]**2)*Du[:,1,2]
        state += DP[:,2]*u[:,1]
        state += (gamma-2)*(u[:,0]*B[:,0] + u[:,1]*B[:,1])*(DB[:,1,2])
        #state = DP[:,0] + u[:,0]*DP[:,1] + u[:,1]*DP[:,2]
        #state+= (gamma-2)*(u[:,0]*B[:,0] + u[:,1]*B[:,1])*(DB[:,0,1] + DB[:,1,2])
        #state+= (gamma*P_gas+B[:,1]**2)*Du[:,0,1] - B[:,0]*B[:,1]*(Du[:,0,2])
        #state+= (gamma*P_gas+B[:,0]**2)*Du[:,1,2] - B[:,0]*B[:,1]*(Du[:,1,1])
        #    -gamma*rho**(-gamma-1)*(Drho[:,0] + Drho[:,1]*u[:,0] + Drho[:,2]*u[:,1])
        
        
        return state
    
    def calc_state_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        state_pde = self.calc_state(x_pde)
        return self.loss_container(state_pde, torch.zeros_like(state_pde))

    def eval_state_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        state_pde = self.calc_state(x_pde)
        return self.loss_container(state_pde, torch.zeros_like(state_pde))

    def calc_ind(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)

        rho = y_pred[:,0]
        u = y_pred[:,1:3]
        P = y_pred[:,3]
        B = y_pred[:,4:6]

        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]
        DP = Dy_pred[:,3]
        DB = Dy_pred[:,4:6]

        ind_x = DB[:,0,0] - B[:,1]*Du[:,0,2] + B[:,0]*Du[:,1,2] + u[:,1]*DB[:,0,2] - u[:,0]*DB[:,1,2]
        ind_y = DB[:,1,0] + B[:,1]*Du[:,0,1] - B[:,0]*Du[:,1,1] - u[:,1]*DB[:,0,1] + u[:,0]*DB[:,1,1]
        
        return torch.stack([ind_x, ind_y], dim=1)

    def calc_ind_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        ind_pde = self.calc_ind(x_pde)
        return self.loss_container(ind_pde, torch.zeros_like(ind_pde))

    def eval_ind_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        ind_pde = self.calc_ind(x_pde)
        return self.loss_container(ind_pde, torch.zeros_like(ind_pde))

    def loss_fn(self,
                x_pde:torch.Tensor,
                x_ic:torch.Tensor, y_ic:torch.Tensor,
                **kwargs
    ) -> torch.Tensor:
        
        # Compute the PDE loss
        cont_loss = self.calc_cont_loss(x_pde)
        mom_loss = self.calc_mom_loss(x_pde)
        state_loss = self.calc_state_loss(x_pde)
        ind_loss = self.calc_ind_loss(x_pde)

        # Compute the initial loss
        ic_loss = self.calc_ic_loss(x_ic, y_ic)

        # Total final loss
        tot_loss = self.cont_weight*cont_loss + self.mom_weight*mom_loss + self.state_weight*state_loss +\
            self.ind_weight*ind_loss + self.ic_weight*ic_loss

        return tot_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor,
                    **kwargs):
        # Get the prediction
        ic_loss = self.eval_ic_loss(x_ic, y_ic)
                
        # Now calculate the error wrt the true pdeution
        y_pred_final = self.forward(x_pde)
        y_loss = self.loss_container(y_pred_final, y_pde)


        cont_loss = self.eval_cont_loss(x_pde)
        mom_loss = self.eval_mom_loss(x_pde)
        state_loss = self.eval_state_loss(x_pde)
        ind_loss = self.eval_ind_loss(x_pde)

        tot_loss = ic_loss + cont_loss + mom_loss + state_loss + ind_loss

        return cont_loss, mom_loss, state_loss, ind_loss, torch.zeros_like(y_loss), y_loss, ic_loss, torch.zeros_like(y_loss), tot_loss
    
        

    def evaluate_consistency(self, x_pde:torch.Tensor, **kwargs):
        # Get the prediction
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        
        rho = y_pred[:,0]
        u = y_pred[:,1:3]
        P = y_pred[:,3]
        B = y_pred[:,4:6]

        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]
        DP = Dy_pred[:,3]
        DB = Dy_pred[:,4:6]

        cont_pde = Drho[:,0] + rho*(Du[:,0,1] + Du[:,1,2]) + Drho[:,1]*u[:,0] + Drho[:,2]*u[:,1]

        mom_pde_x = rho*(Du[:,0,0] + Du[:,0,1]*u[:,0] + Du[:,0,2]*u[:,1])
        mom_pde_x += (DP[:,1] - 2*B[:,0]*DB[:,0,1] - B[:,1]*DB[:,0,2] - B[:,0]*DB[:,1,2])
        #mom_pde_x = (Du[:,0,0] + Du[:,0,1]*u[:,0] + Du[:,0,2]*u[:,1])
        #mom_pde_x += (DP[:,1]/rho - 2*B[:,0]/rho*DB[:,0,1] - B[:,1]/rho*DB[:,0,2] - B[:,0]/rho*DB[:,1,2])
        
        mom_pde_y = rho*(Du[:,1,0] + Du[:,1,1]*u[:,0] + Du[:,1,2]*u[:,1])
        mom_pde_y += (DP[:,2] - 2*B[:,1]*DB[:,1,2] - B[:,1]*DB[:,0,1] - B[:,0]*DB[:,1,1])
        #mom_pde_y = (Du[:,1,0] + Du[:,1,1]*u[:,0] + Du[:,1,2]*u[:,1])
        #mom_pde_y += (DP[:,2]/rho - 2*B[:,1]/rho*DB[:,1,2] - B[:,0]/rho*DB[:,1,1] - B[:,1]/rho*DB[:,0,1])

        mom_pde = torch.stack([mom_pde_x, mom_pde_y], dim=1)

        P_gas = P - 0.5 * (B[:,0]**2 + B[:,1]**2)

        state = DP[:,0] + (gamma*P_gas + B[:,1]**2)*Du[:,0,1]
        state -= B[:,0]*B[:,1]*(Du[:,1,1])
        state += DP[:,1]*u[:,0]
        state += (gamma-2)*(u[:,0]*B[:,0] + u[:,1]*B[:,1])*(DB[:,0,1])
        state -= B[:,0]*B[:,1]*(Du[:,0,2])
        state += (gamma*P_gas + B[:,0]**2)*Du[:,1,2]
        state += DP[:,2]*u[:,1]
        state += (gamma-2)*(u[:,0]*B[:,0] + u[:,1]*B[:,1])*(DB[:,1,2])

        ind_x = DB[:,0,0] - B[:,1]*Du[:,0,2] + B[:,0]*Du[:,1,2] + u[:,1]*DB[:,0,2] - u[:,0]*DB[:,1,2]
        ind_y = DB[:,1,0] + B[:,1]*Du[:,0,1] - B[:,0]*Du[:,1,1] - u[:,1]*DB[:,0,1] + u[:,0]*DB[:,1,1]
        ind_pde =  torch.stack([ind_x, ind_y], dim=1)
        
        gauss = DB[:,0,1] + DB[:,1,2]
        return torch.abs(cont_pde), torch.abs(mom_pde), torch.abs(state), torch.abs(ind_pde), torch.abs(gauss)