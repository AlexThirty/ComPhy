
import torch.utils
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch.func import vmap, jacrev, jacfwd, hessian
import numpy as np

gamma = 5./3.

class SoftAbs(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(SoftAbs, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.abs(x)>1, x-0.5, 0.5*x**2)

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class QuadPINN(torch.nn.Module):
    def __init__(self,
                 alignment_mode:str,
                 hidden_units: list=[32, 32, 32],
                 cont_weight:float=1.,
                 mom_weight:float=1.,
                 state_weight:float=1.,
                 ind_weight:float=1.,
                 gauss_weight:float=1.,
                 alignment_weight:float=1.,
                 ic_weight:float=1.,
                 bc_weight:float=1.,
                 ind_activation:nn.Module=nn.GELU(),
                 mom_activation:nn.Module=nn.GELU(),
                 state_activation:nn.Module=nn.GELU(),
                 cont_activation:nn.Module=nn.GELU(),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self.device = device
        
        
        self.in_dim = 5
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.cont_weight = cont_weight
        self.mom_weight = mom_weight
        self.state_weight = state_weight
        self.ind_weight = ind_weight
        self.gauss_weight = gauss_weight
        self.hidden_units = hidden_units
        self.alignment_weight = alignment_weight
        self.alignment_mode = alignment_mode
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 
        
        # indergence free network
        ind_out_dim = 4
        self.ind_out_dim = ind_out_dim
        self.ind_hidden_units = [self.in_dim] + hidden_units
    
        ind_net = nn.Sequential()
        for i in range(len(self.ind_hidden_units)-1):
            ind_net.add_module(f'ind_lin{i}', nn.Linear(self.ind_hidden_units[i], self.ind_hidden_units[i+1]))
            ind_net.add_module(f'ind_act{i}', ind_activation)
        ind_net.add_module(f'ind_lin{len(self.ind_hidden_units)-1}', nn.Linear(self.ind_hidden_units[-1], self.ind_out_dim))
        
        self.ind_net = ind_net.to(self.device)
        
        # Momentum equation network # TODO: if it does not work well, split in two
        mom_out_dim = 6
        self.mom_out_dim = mom_out_dim
        self.mom_hidden_units = [self.in_dim] + hidden_units
        mom_net = nn.Sequential()
        for i in range(len(self.mom_hidden_units)-1):
            mom_net.add_module(f'mom_lin{i}', nn.Linear(self.mom_hidden_units[i], self.mom_hidden_units[i+1]))
            mom_net.add_module(f'mom_act{i}', mom_activation)
        mom_net.add_module(f'mom_lin{len(self.mom_hidden_units)-1}', nn.Linear(self.mom_hidden_units[-1], mom_out_dim))
        
        self.mom_net = mom_net.to(self.device)
        
        # Momentum equation network # TODO: if it does not work well, split in two
        state_out_dim = 5
        self.state_out_dim = state_out_dim
        self.state_hidden_units = [self.in_dim] + hidden_units
        state_net = nn.Sequential()
        for i in range(len(self.state_hidden_units)-1):
            state_net.add_module(f'state_lin{i}', nn.Linear(self.state_hidden_units[i], self.state_hidden_units[i+1]))
            state_net.add_module(f'state_act{i}', state_activation)
        state_net.add_module(f'state_lin{len(self.state_hidden_units)-1}', nn.Linear(self.state_hidden_units[-1], state_out_dim))
        
        self.state_net = state_net.to(self.device)
        
        # Momentum equation network # TODO: if it does not work well, split in two
        cont_out_dim = 2
        self.cont_out_dim = cont_out_dim
        self.cont_hidden_units = [self.in_dim] + hidden_units
        cont_net = nn.Sequential()
        for i in range(len(self.cont_hidden_units)-1):
            cont_net.add_module(f'cont_lin{i}', nn.Linear(self.cont_hidden_units[i], self.cont_hidden_units[i+1]))
            cont_net.add_module(f'cont_act{i}', cont_activation)
        cont_net.add_module(f'cont_lin{len(self.cont_hidden_units)-1}', nn.Linear(self.cont_hidden_units[-1], cont_out_dim))
        
        self.cont_net = cont_net.to(self.device)
        # Print the number of parameters
        ind_params = sum(p.numel() for p in self.ind_net.parameters())
        mom_params = sum(p.numel() for p in self.mom_net.parameters())
        state_params = sum(p.numel() for p in self.state_net.parameters())
        cont_params = sum(p.numel() for p in self.cont_net.parameters())
        total_params = ind_params + mom_params + state_params + cont_params
        print(f"Number of parameters in ind_net: {ind_params}")
        print(f"Number of parameters in mom_net: {mom_params}")
        print(f"Number of parameters in state_net: {state_params}")
        print(f"Number of parameters in cont_net: {cont_params}")
        print(f"Total number of parameters: {total_params}")
        
        self.device = device
        
    def embed(self, x:torch.Tensor):
        c = 2*np.pi
        x = x.reshape((-1,3))
        return torch.cat([x[:,:1], torch.cos(c*x[:,1:2]), torch.sin(c*x[:,1:2]), torch.cos(c*x[:,2:3]), torch.sin(c*x[:,2:3])], dim=1).squeeze(0)
                #torch.cos(2*c*x[:,1:2]), torch.sin(2*c*x[:,1:2]), torch.cos(2*c*x[:,2:3]), torch.sin(2*c*x[:,2:3])], dim=1).squeeze(0)


    def forward(self, tx, return_final=False):
        # Get the predictions
        tx = self.embed(tx)
        ind_out = self.ind_net(tx).reshape((-1,4))
        mom_out = self.mom_net(tx).reshape((-1,6))
        state_out = self.state_net(tx).reshape((-1,5))
        cont_out = self.cont_net(tx).reshape((-1,2))
        if return_final:
            return mom_out

        #return torch.cat((self.positivizer(mom_out[:,:1]), mom_out[:,1:3], self.positivizer(mom_out[:,3:4]), mom_out[:,4:6], state_out[:,0:2], self.positivizer(state_out[:,2:3]), state_out[:,3:5], ind_out, self.positivizer(cont_out[:,:1]), cont_out[:,1:]), dim=1).squeeze(0)
        return torch.cat((mom_out, state_out, ind_out, cont_out), dim=1).squeeze(0)



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
        y_mom_pred = y_pred[:,:6]
        y_state_pred = y_pred[:,6:11]
        y_ind_pred = y_pred[:,11:15]
        y_cont_pred = y_pred[:,15:]
        
        Dy_pred = vmap(jacrev(self.forward))(x_pde)
        Dy_mom_pred = Dy_pred[:,:6]
        Dy_state_pred = Dy_pred[:,6:11]
        Dy_ind_pred  = Dy_pred[:,11:15]
        Dy_cont_pred = Dy_pred[:,15:]
        
        # Now we impose consistency between the branches
        if self.alignment_mode ==  'DERL':
            # Match the derivatives between the branches
            alignment_loss = self.alignment_loss_container(Dy_ind_pred - torch.cat([Dy_mom_pred[:,1:3], Dy_mom_pred[:,4:6]], dim=1), torch.zeros_like(Dy_ind_pred))
            alignment_loss += self.alignment_loss_container(Dy_state_pred - Dy_mom_pred[:,1:], torch.zeros_like(Dy_state_pred))
            alignment_loss += self.alignment_loss_container(Dy_cont_pred - Dy_mom_pred[:,-2:], torch.zeros_like(Dy_cont_pred))
            #alignment_loss += self.alignment_loss_container(Dy_ind_pred - torch.cat([Dy_state_pred[:,0:2], Dy_state_pred[:,3:5]], dim=1), torch.zeros_like(Dy_ind_pred))
            #alignment_loss += self.alignment_loss_container(Dy_ind_pred[:,:2] - Dy_cont_pred[:,1:3], torch.zeros_like(Dy_ind_pred[:,:2]))
            #alignment_loss += self.alignment_loss_container(Dy_state_pred[:,:2] - Dy_cont_pred[:,1:3], torch.zeros_like(Dy_state_pred[:,:2]))
        elif self.alignment_mode == 'OUTL':
            # Match the outputs between the branches
            alignment_loss = self.alignment_loss_container(y_ind_pred - torch.cat([y_mom_pred[:,1:3], y_mom_pred[:,4:6]], dim=1), torch.zeros_like(y_ind_pred))
            alignment_loss += self.alignment_loss_container(y_state_pred - y_mom_pred[:,1:], torch.zeros_like(y_state_pred))
            alignment_loss += self.alignment_loss_container(y_cont_pred - y_mom_pred[:,-2:], torch.zeros_like(y_cont_pred))
            #alignment_loss += self.alignment_loss_container(y_ind_pred - torch.cat([y_state_pred[:,0:2], y_state_pred[:,3:5]], dim=1), torch.zeros_like(y_ind_pred))
            #alignment_loss += self.alignment_loss_container(y_ind_pred[:,:2] - y_cont_pred[:,1:3], torch.zeros_like(y_ind_pred[:,:2]))
            #alignment_loss += self.alignment_loss_container(y_state_pred[:,:2] - y_cont_pred[:,1:3], torch.zeros_like(y_state_pred[:,:2]))
        elif self.alignment_mode == 'SOB':
            # It is the sum of the other two
            alignment_loss = self.alignment_loss_container(Dy_ind_pred - torch.cat([Dy_mom_pred[:,1:3], Dy_mom_pred[:,4:6]], dim=1), torch.zeros_like(Dy_ind_pred))
            alignment_loss += self.alignment_loss_container(y_ind_pred - torch.cat([y_mom_pred[:,1:3], y_mom_pred[:,4:6]], dim=1), torch.zeros_like(y_ind_pred))
            alignment_loss += self.alignment_loss_container(Dy_state_pred - Dy_mom_pred[:,1:], torch.zeros_like(Dy_state_pred))
            alignment_loss += self.alignment_loss_container(y_state_pred - y_mom_pred[:,1:], torch.zeros_like(y_state_pred))
            alignment_loss += self.alignment_loss_container(Dy_cont_pred - Dy_mom_pred[:,-2:], torch.zeros_like(Dy_cont_pred))
            alignment_loss += self.alignment_loss_container(y_cont_pred - y_mom_pred[:,-2:], torch.zeros_like(y_cont_pred))
            #alignment_loss += self.alignment_loss_container(Dy_ind_pred - torch.cat([Dy_state_pred[:,0:2], Dy_state_pred[:,3:5]],dim=1), torch.zeros_like(Dy_ind_pred))
            #alignment_loss += self.alignment_loss_container(Dy_ind_pred[:,:2] - Dy_cont_pred[:,1:3], torch.zeros_like(Dy_ind_pred[:,:2]))
            #alignment_loss += self.alignment_loss_container(Dy_state_pred[:,:2] - Dy_cont_pred[:,1:3], torch.zeros_like(Dy_state_pred[:,:2]))
            #alignment_loss += self.alignment_loss_container(y_ind_pred - torch.cat([y_state_pred[:,0:2], y_state_pred[:,3:5]], dim=1), torch.zeros_like(y_ind_pred))
            #alignment_loss += self.alignment_loss_container(y_ind_pred[:,:2] - y_cont_pred[:,1:3], torch.zeros_like(y_ind_pred[:,:2]))
            #alignment_loss += self.alignment_loss_container(y_state_pred[:,:2] - y_cont_pred[:,1:3], torch.zeros_like(y_state_pred[:,:2]))
        return alignment_loss

    def calc_ic_loss(self,
                     x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_ic)
        
        y_mom_pred = y_pred[:,:6]
        y_state_pred = y_pred[:,6:11]
        y_ind_pred = y_pred[:,11:15]
        y_cont_pred = y_pred[:,15:]
        
        return self.loss_container(y_mom_pred, y_ic) + self.loss_container(y_state_pred, y_ic[:,1:]) +\
            self.loss_container(y_ind_pred, torch.cat([y_ic[:,1:3], y_ic[:,4:6]], dim=1)) + self.loss_container(y_cont_pred, y_ic[:,-2:])

    def calc_ic(self, x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return y_ic_pred

    def eval_ic_loss(self,
                     x_ic:torch.Tensor, y_ic:torch.Tensor, **kwargs) -> torch.Tensor:
        y_ic_pred = self.forward(x_ic)
        return self.loss_container(y_ic_pred[:,:6], y_ic)
        
    def calc_cont(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)[:,:3]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:3]

        rho = y_pred[:,0]
        u = y_pred[:,1:3]

        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]

        cont_pde = Drho[:,0] + rho*(Du[:,0,1] + Du[:,1,2]) + Drho[:,1]*u[:,0] + Drho[:,2]*u[:,1]
        return cont_pde

    def calc_cont_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        cont_pde = self.calc_cont(x_pde)
        return self.loss_container(cont_pde, torch.zeros_like(cont_pde))

    def eval_cont_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)[:,:6]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:6]
        
        rho = y_pred[:,0]
        u = y_pred[:,1:3]


        Drho = Dy_pred[:,0]
        Du = Dy_pred[:,1:3]


        cont_pde = Drho[:,0] + rho*(Du[:,0,1] + Du[:,1,2]) + Drho[:,1]*u[:,0] + Drho[:,2]*u[:,1]
        return self.loss_container(cont_pde, torch.zeros_like(cont_pde))

    def calc_mom(
        self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        y_pred = self.forward(x_pde)[:,:6]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,:6]
        
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
        y_pred = self.forward(x_pde)[:,6:11]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,6:11]
        
        u = y_pred[:,0:2]
        P = y_pred[:,2]
        B = y_pred[:,3:5]

        P_gas = P - 0.5 * (B[:,0]**2 + B[:,1]**2)

        Du = Dy_pred[:,0:2]
        DP = Dy_pred[:,2]
        DB = Dy_pred[:,3:5]

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
        
        return self.loss_container(state, torch.zeros_like(state))


    def eval_state_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        state_pde = self.calc_state(x_pde)
        return self.loss_container(state_pde, torch.zeros_like(state_pde))
    
    def calc_ind(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)[:,11:15]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,11:15]

        u = y_pred[:,0:2]
        B = y_pred[:,2:4]

        Du = Dy_pred[:,0:2]
        DB = Dy_pred[:,2:4]

        ind_x = DB[:,0,0] - B[:,1]*Du[:,0,2] + B[:,0]*Du[:,1,2] + u[:,1]*DB[:,0,2] - u[:,0]*DB[:,1,2]
        ind_y = DB[:,1,0] + B[:,1]*Du[:,0,1] - B[:,0]*Du[:,1,1] - u[:,1]*DB[:,0,1] + u[:,0]*DB[:,1,1]
        
        return torch.stack([ind_x, ind_y], dim=1)

    def eval_ind_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
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
        
        ind = torch.stack([ind_x, ind_y], dim=1)
        return self.loss_container(ind, torch.zeros_like(ind))

    def calc_ind_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        ind_pde = self.calc_ind(x_pde)
        return self.loss_container(ind_pde, torch.zeros_like(ind_pde))

    def calc_gauss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)[:,15:]
        Dy_pred = vmap(jacrev(self.forward))(x_pde)[:,15:]

        B = y_pred[:,:2]
        DB = Dy_pred[:,:2]

        gauss = DB[:,0,1] + DB[:,1,2]
        return gauss

    
    def calc_gauss_loss(self,
        x_pde:torch.Tensor, **kwargs) -> torch.Tensor:

        gauss_pde = self.calc_gauss(x_pde)
        return self.loss_container(gauss_pde, torch.zeros_like(gauss_pde))

    def eval_gauss_loss(self, x_pde:torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred = self.forward(x_pde)
        Dy_pred = vmap(jacrev(self.forward))(x_pde)

        B = y_pred[:,4:6]
        DB = Dy_pred[:,4:6]

        gauss = DB[:,0,1] + DB[:,1,2]
        return self.loss_container(gauss, torch.zeros_like(gauss))

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
        gauss_loss = self.calc_gauss_loss(x_pde)
        
        alignment_loss = self.calc_alignment_loss(x_pde)

        # Compute the initial loss
        ic_loss = self.calc_ic_loss(x_ic, y_ic)

        # Total final loss
        tot_loss = self.cont_weight*cont_loss + self.mom_weight*mom_loss + self.state_weight*state_loss +\
            self.ind_weight*ind_loss + self.gauss_weight*gauss_loss + self.ic_weight*ic_loss + self.alignment_weight*alignment_loss

        return tot_loss

    def eval_losses(self,
                    x_pde:torch.Tensor, y_pde:torch.Tensor,
                    x_ic:torch.Tensor, y_ic:torch.Tensor,
                    **kwargs):
        # Get the prediction
        ic_loss = self.eval_ic_loss(x_ic, y_ic)
                
        # Now calculate the error wrt the true pdeution
        y_pred_final = self.forward(x_pde)[:,:6]
        y_loss = self.loss_container(y_pred_final, y_pde)


        cont_loss = self.eval_cont_loss(x_pde)
        mom_loss = self.eval_mom_loss(x_pde)
        state_loss = self.eval_state_loss(x_pde)
        ind_loss = self.eval_ind_loss(x_pde)
        gauss_loss = self.eval_gauss_loss(x_pde)
        
        alignment_loss = self.calc_alignment_loss(x_pde)

        tot_loss = ic_loss + cont_loss + mom_loss + state_loss + ind_loss + gauss_loss + alignment_loss

        return cont_loss, mom_loss, state_loss, ind_loss, gauss_loss, y_loss, ic_loss, alignment_loss, tot_loss


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