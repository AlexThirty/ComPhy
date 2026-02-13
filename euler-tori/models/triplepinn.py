import torch.utils
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch.func import vmap, jacrev, jacfwd, hessian
import numpy as np

class modSoftplus(nn.Module):
    def __init__(self, beta:float=1., threshold:float=20.) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.nn.Softplus(beta=self.beta, threshold=self.threshold)(x*self.beta)/self.beta

class SinActivation(nn.Module):
    def __init__(self, beta:float=1.) -> None:
        super().__init__()
        self.beta = beta
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.sin(self.beta*x)

class TriplePINN(torch.nn.Module):
    def __init__(self,
                 div_hidden_units:list,
                 inc_hidden_units:list,
                 mom_hidden_units:list,
                 div_weight:float=1.,
                 mom_weight:float=1.,
                 init_weight:float=1.,
                 inc_weight:float=1.,
                 alignment_weight:float=1.,
                 lr:float=1e-3,
                 div_activation:nn.Module=nn.Tanh(),
                 inc_activation:nn.Module=nn.Tanh(),
                 mom_activation:nn.Module=nn.Tanh(),
                 device: str='cuda:0',
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        
        self.in_dim = 5
        self.init_weight = init_weight
        self.inc_weight = inc_weight
        self.mom_weight = mom_weight
        self.div_weight = div_weight
        self.alignment_weight = alignment_weight
        self.div_hidden_units = div_hidden_units
        
        self.loss_container = torch.nn.MSELoss(reduction='mean') 
        self.alignment_loss_container = torch.nn.MSELoss(reduction='mean') 

        
        # Divergence free network
        div_out_dim = 3
        self.div_out_dim = div_out_dim
        self.div_hidden_units = [self.in_dim] + div_hidden_units
    
        div_net = nn.Sequential()
        for i in range(len(self.div_hidden_units)-1):
            div_net.add_module(f'div_lin{i}', nn.Linear(self.div_hidden_units[i], self.div_hidden_units[i+1]))
            div_net.add_module(f'div_act{i}', div_activation)
        div_net.add_module(f'div_lin{len(self.div_hidden_units)-1}', nn.Linear(self.div_hidden_units[-1], self.div_out_dim))
        
        self.div_net = div_net.to(self.device)
        
        inc_out_dim = 2
        self.inc_out_dim = inc_out_dim
        self.inc_hidden_units = [self.in_dim] + inc_hidden_units
        
        inc_net = nn.Sequential()
        for i in range(len(self.inc_hidden_units)-1):
            inc_net.add_module(f'inc_lin{i}', nn.Linear(self.inc_hidden_units[i], self.inc_hidden_units[i+1]))
            inc_net.add_module(f'inc_act{i}', inc_activation)
        inc_net.add_module(f'inc_lin{len(self.inc_hidden_units)-1}', nn.Linear(self.inc_hidden_units[-1], self.inc_out_dim))
        
        self.inc_net = inc_net.to(self.device)
        
        mom_net = nn.Sequential()
        mom_hidden_units = [self.in_dim] + mom_hidden_units
        mom_out_dim = 4
        self.mom_hidden_units = mom_hidden_units
        self.mom_out_dim = mom_out_dim
        for i in range(len(mom_hidden_units)-1):
            mom_net.add_module(f'mom_lin{i}', nn.Linear(mom_hidden_units[i], mom_hidden_units[i+1]))
            mom_net.add_module(f'mom_act{i}', mom_activation)
        mom_net.add_module(f'mom_lin{len(mom_hidden_units)-1}', nn.Linear(mom_hidden_units[-1], mom_out_dim))
        
        self.mom_net = mom_net.to(self.device)
        # Save the optimizer
        self.lr = lr
        
        self.device = device
    
    def embed(self, x:torch.Tensor):
        c = 2*np.pi
        return torch.cat([x[:,:1], torch.cos(c*x[:,1:2]), torch.sin(c*x[:,1:2]), torch.cos(c*x[:,2:3]), torch.sin(c*x[:,2:3])], dim=1)
    
    def embed_single(self, x:torch.Tensor):
        c = 2*np.pi
        return torch.cat([x[:1], torch.cos(c*x[1:2]), torch.sin(c*x[1:2]), torch.cos(c*x[2:3]), torch.sin(c*x[2:3])], dim=0)
    
    def forward_single(self, tx: torch.Tensor, return_final:bool=False):
        tx_in = self.embed_single(tx)
        # Now get the vector
        div_out = self.div_net(tx_in)
        div_out[0] += 2.
        div_out[1:3] += 1e-1
        
        inc_out = self.inc_net(tx_in)
        inc_out += (1e-1)/2.
        
        mom_out = self.mom_net(tx_in)
        mom_out[0] += 2.
        mom_out[1:3] += 1e-1
        if return_final:
            return mom_out
        return torch.concat([mom_out, div_out, inc_out], dim=0)
    
    def forward(self, tx: torch.Tensor, return_final:bool=False):
        tx_in = self.embed(tx)
        
        div_out = self.div_net(tx_in)
        div_out[:,0] += 2.
        div_out[:,1:3] += 1e-1
        
        inc_out = self.inc_net(tx_in)
        inc_out += (1e-1)/2.
        
        mom_out = self.mom_net(tx_in)
        mom_out[:,0] += 2.
        mom_out[:,1:3] += 1e-1
        
        if return_final:
            return mom_out
        return torch.column_stack((mom_out, div_out, inc_out))
    
    def forward_u_single(self, tx: torch.Tensor):
        tx_in = self.embed_single(tx)

        mom_out = self.mom_net(tx_in)

        mom_out[0] += 2.
        mom_out[1:3] += 1e-1
        return mom_out[1:3]/mom_out[0]

    
    def loss_fn(self, 
                x_pde:torch.Tensor,
                x_init:torch.Tensor, y_init:torch.Tensor,
                alignment_mode:str='DERL'
        ) -> torch.Tensor:
        # Get the prediction (rho, rhou, rhov)
        y_pred = self.forward(x_pde)
        y_mom = y_pred[:,:4]
        y_div = y_pred[:,4:7]
        y_inc = y_pred[:,7:]
                
        rho_mom = y_mom[:,0]
        rhou_mom = y_mom[:,1:3]
        p_mom = y_mom[:,-1]        
        
        rho_div = y_div[:,0]
        rhou_div = y_div[:,1:3]        
        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        
        Dy_mom = Dy_pred[:,:4]
        Dy_div = Dy_pred[:,4:7]
        Dy_inc = Dy_pred[:,7:]
        
        Drho_mom = Dy_mom[:,0]
        Drhou_mom = Dy_mom[:,1:3]
        Dp_mom = Dy_mom[:,-1]
        
        Drho_div = Dy_div[:,0]
        Drhou_div = Dy_div[:,1:3]
        
        
        # Get the divergence (continuity) loss
        div_pde = vmap(torch.trace)(Dy_div[:,:3])
        div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        # Get the momentum loss
        term_1 = rho_mom.unsqueeze(1)**2 * Drhou_mom[:,:,0]
        term_2 = rho_mom.unsqueeze(1) * (Drho_mom[:,0]).unsqueeze(1) * rhou_mom
        term_3 = rho_mom.unsqueeze(1) * torch.einsum('bij, bj -> bi', Drhou_mom[:,:,1:], rhou_mom)
        term_4 = torch.einsum('bij, bj -> bi', torch.einsum('bi, bj -> bji', Drho_mom[:,1:], rhou_mom), rhou_mom)
        term_5 = rho_mom.unsqueeze(1)**2 * Dp_mom[:,1:]
        
        mom_pde = term_1 - term_2 + term_3 - term_4 + term_5
        
        mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
        
        # Incompressibility loss
        inc_pde = vmap(torch.trace)(Dy_inc[:,:,1:3])
        inc_loss = self.loss_container(inc_pde, torch.zeros_like(inc_pde))
        
        
        y_init_pred = self.forward(x_init)
        y_init_mom = y_init_pred[:,:3]
        y_init_div = y_init_pred[:,4:7]
        y_init_inc = y_init_pred[:,7:]
        init_loss = self.loss_container(y_init_mom, y_init) + self.loss_container(y_init_div, y_init) + self.loss_container(y_init_inc, y_init[:,1:3]/y_init[:,:1])
        
        
        # TODO try the other way around, here we calculate the u of the divergence free network
        Du_mom = vmap(jacrev(self.forward_u_single))(x_pde)
        u_mom = rhou_div/rho_div.unsqueeze(1)
        
        if alignment_mode == 'DERL':
            alignment_loss = self.loss_container(Du_mom - Dy_inc, torch.zeros_like(Du_mom))
            alignment_loss+= self.loss_container(Dy_mom[:,:3] - Dy_div, torch.zeros_like(Dy_mom[:,:3]))
        elif alignment_mode == 'OUTL':
            alignment_loss = self.loss_container(u_mom - y_inc, torch.zeros_like(u_mom))
            alignment_loss+= self.loss_container(y_mom[:,:3] - y_div, torch.zeros_like(y_mom[:,:3]))
        elif alignment_mode == 'SOB':
            alignment_loss = self.loss_container(Du_mom - Dy_inc, torch.zeros_like(Du_mom)) + self.loss_container(u_mom - y_inc, torch.zeros_like(u_mom))
            alignment_loss+= self.loss_container(Dy_mom[:,:3] - Dy_div, torch.zeros_like(Dy_mom[:,:3])) + self.loss_container(y_mom[:,:3] - y_div, torch.zeros_like(y_mom[:,:3]))
        else:
            raise ValueError('alignment mode not recognized')
    
        return init_loss*self.init_weight + self.mom_weight*mom_loss + self.alignment_weight*alignment_loss + self.div_weight*div_loss + self.inc_weight*inc_loss
    
    def eval_losses(self, 
                    x_pde:torch.Tensor,
                    x_init:torch.Tensor, y_init:torch.Tensor,
                    x_sol:torch.Tensor, y_sol:torch.Tensor,
                    step:int, alignment_mode:str='DERL') -> torch.Tensor:
        
        
        # Get the prediction (rho, rhou, rhov)
        y_pred = self.forward(x_pde)
        y_mom = y_pred[:,:4]
        y_div = y_pred[:,4:7]
        y_inc = y_pred[:,7:]
        
        rho_mom = y_mom[:,0]
        rhou_mom = y_mom[:,1:3]
        p_mom = y_mom[:,-1]
        
        rho_div = y_div[:,0]
        rhou_div = y_div[:,1:3]
        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        
        Dy_mom = Dy_pred[:,:4]
        Dy_div = Dy_pred[:,4:7]
        Dy_inc = Dy_pred[:,7:]
        
        Drho_mom = Dy_mom[:,0]
        Drhou_mom = Dy_mom[:,1:3]
        Dp_mom = Dy_mom[:,-1]
        
        Drho_div = Dy_div[:,0]
        Drhou_div = Dy_div[:,1:3]
        
        # Get the divergence (continuity) loss
        div_pde = vmap(torch.trace)(Dy_div[:,:3])
        div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
        
        # Get the momentum loss
        term_1 = rho_mom.unsqueeze(1)**2 * Drhou_mom[:,:,0]
        term_2 = rho_mom.unsqueeze(1) * (Drho_mom[:,0]).unsqueeze(1) * rhou_mom
        term_3 = rho_mom.unsqueeze(1) * torch.einsum('bij, bj -> bi', Drhou_mom[:,:,1:], rhou_mom)
        term_4 = torch.einsum('bij, bj -> bi', torch.einsum('bi, bj -> bji', Drho_mom[:,1:], rhou_mom), rhou_mom)
        term_5 = rho_mom.unsqueeze(1)**2 * Dp_mom[:,1:]
        
        mom_pde = term_1 - term_2 + term_3 - term_4 + term_5
        
        mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
        
        # Incompressibility loss
        inc_pde = vmap(torch.trace)(Dy_inc[:,:,1:3])
        inc_loss = self.loss_container(inc_pde, torch.zeros_like(inc_pde))
        
        y_init_pred = self.forward(x_init)
        y_init_mom = y_init_pred[:,:3]
        y_init_div = y_init_pred[:,4:7]
        y_init_inc = y_init_pred[:,7:]
        init_loss = self.loss_container(y_init_mom, y_init) + self.loss_container(y_init_div, y_init) + self.loss_container(y_init_inc, y_init[:,1:3]/y_init[:,:1])
        
        # TODO try the other way around, here we calculate the u of the divergence free network
        Du_mom = vmap(jacrev(self.forward_u_single))(x_pde)
        u_mom = rhou_div/rho_div.unsqueeze(1)
        
        if alignment_mode == 'DERL':
            alignment_loss = self.loss_container(Du_mom - Dy_inc, torch.zeros_like(Du_mom))
            alignment_loss+= self.loss_container(Dy_mom[:,:3] - Dy_div, torch.zeros_like(Dy_mom[:,:3]))
        elif alignment_mode == 'OUTL':
            alignment_loss = self.loss_container(u_mom - y_inc, torch.zeros_like(u_mom))
            alignment_loss+= self.loss_container(y_mom[:,:3] - y_div, torch.zeros_like(y_mom[:,:3]))
        elif alignment_mode == 'SOB':
            alignment_loss = self.loss_container(Du_mom - Dy_inc, torch.zeros_like(Du_mom)) + self.loss_container(u_mom - y_inc, torch.zeros_like(u_mom))
            alignment_loss+= self.loss_container(Dy_mom[:,:3] - Dy_div, torch.zeros_like(Dy_mom[:,:3])) + self.loss_container(y_mom[:,:3] - y_div, torch.zeros_like(y_mom[:,:3]))
        else:
            raise ValueError('alignment mode not recognized')
        
        inc_pde = vmap(torch.trace)(Du_mom[:,:,1:3])
        inc_loss = self.loss_container(inc_pde, torch.zeros_like(inc_pde))
         
        y_sol_pred = self.forward(x_sol)[:,:3]
        y_loss = self.alignment_loss_container(y_sol_pred, y_sol)
        
        tot_loss_val = div_loss + inc_loss + mom_loss + init_loss + alignment_loss
        return step, div_loss, mom_loss, inc_loss, y_loss, init_loss, alignment_loss, tot_loss_val
    
    def forward_single_final(self, tx: torch.Tensor):
        return self.forward_single(tx, return_final=True)
    
    def evaluate_consistency(self, x_pde:torch.Tensor):
        Du = vmap(jacrev(self.forward_u_single))(x_pde)
        Dy = vmap(jacrev(self.forward_single_final))(x_pde)
        Drho = Dy[:,0]
        Drhou = Dy[:,1:3]
        Dp = Dy[:,-1]
        
        
        y = self.forward(x_pde, return_final=True)
        rho = y[:,0]
        rhou = y[:,1:3]
        p = y[:,-1]
        
        
        inc_pde = vmap(torch.trace)(Du[:,:,1:3])
        
        
        term_1 = rho.unsqueeze(1)**2 * Drhou[:,:,0]
        term_2 = rho.unsqueeze(1) * (Drho[:,0]).unsqueeze(1) * rhou
        term_3 = rho.unsqueeze(1) * torch.einsum('bij, bj -> bi', Drhou[:,:,1:], rhou)
        term_4 = torch.einsum('bij, bj -> bi', torch.einsum('bi, bj -> bji', Drho[:,1:], rhou) , rhou)
        term_5 = rho.unsqueeze(1)**2 * Dp[:,1:]        
        
        mom_pde = term_1 - term_2 + term_3 - term_4 + term_5
        
        
        div_pde = vmap(torch.trace)(Dy[:,:3])
        
        return torch.abs(mom_pde), torch.abs(div_pde), torch.abs(inc_pde)
        
        
    def calc_ic_loss(self, x_init:torch.Tensor, y_init:torch.Tensor):
        y_init_pred = self.forward(x_init)
        y_init_mom = y_init_pred[:,:3]
        y_init_div = y_init_pred[:,4:7]
        y_init_inc = y_init_pred[:,7:]
        init_loss = self.loss_container(y_init_mom, y_init) + self.loss_container(y_init_div, y_init) + self.loss_container(y_init_inc, y_init[:,1:3]/y_init[:,:1])
        
        return init_loss

    def calc_inc_loss(self, x_pde:torch.Tensor):        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        Dy_inc = Dy_pred[:,7:]
           
        # Incompressibility loss
        inc_pde = vmap(torch.trace)(Dy_inc[:,:,1:3])
        inc_loss = self.loss_container(inc_pde, torch.zeros_like(inc_pde))
        return inc_loss
    
    def calc_mom_loss(self, x_pde:torch.Tensor):
        
        # Get the prediction (rho, rhou, rhov)
        y_pred = self.forward(x_pde)
        y_mom = y_pred[:,:4]
        y_div = y_pred[:,4:7]
        y_inc = y_pred[:,7:]
        
        rho_mom = y_mom[:,0]
        rhou_mom = y_mom[:,1:3]
        p_mom = y_mom[:,-1]
        
        rho_div = y_div[:,0]
        rhou_div = y_div[:,1:3]
        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        
        Dy_mom = Dy_pred[:,:4]
        Dy_div = Dy_pred[:,4:7]
        Dy_inc = Dy_pred[:,7:]
        
        Drho_mom = Dy_mom[:,0]
        Drhou_mom = Dy_mom[:,1:3]
        Dp_mom = Dy_mom[:,-1]
        
        Drho_div = Dy_div[:,0]
        Drhou_div = Dy_div[:,1:3]
        y_pred = self.forward(x_pde)
        y_div = y_pred[:,:4]
        
        rho_div = y_div[:,0]
        rhou_div = y_div[:,1:3]
        
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)[:,:4]
        
        Drho_div = Dy_pred[:,0]
        Drhou_div = Dy_pred[:,1:3]
        
        # Get the momentum loss
        term_1 = rho_mom.unsqueeze(1)**2 * Drhou_mom[:,:,0]
        term_2 = rho_mom.unsqueeze(1) * (Drho_mom[:,0]).unsqueeze(1) * rhou_mom
        term_3 = rho_mom.unsqueeze(1) * torch.einsum('bij, bj -> bi', Drhou_mom[:,:,1:], rhou_mom)
        term_4 = torch.einsum('bij, bj -> bi', torch.einsum('bi, bj -> bji', Drho_mom[:,1:], rhou_mom), rhou_mom)
        term_5 = rho_mom.unsqueeze(1)**2 * Dp_mom[:,1:]
        
        mom_pde = term_1 - term_2 + term_3 - term_4 + term_5
        
        mom_loss = self.loss_container(mom_pde, torch.zeros_like(mom_pde))
        return mom_loss

    def calc_div_loss(self, x_pde:torch.Tensor):        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        Dy_div = Dy_pred[:,4:7]
        # Get the divergence (continuity) loss
        div_pde = vmap(torch.trace)(Dy_div[:,:3])
        div_loss = self.loss_container(div_pde, torch.zeros_like(div_pde))
        return div_loss

    def calc_align_loss(self, x_pde:torch.Tensor, alignment_mode:str):
        # Get the prediction (rho, rhou, rhov)
        y_pred = self.forward(x_pde)
        y_mom = y_pred[:,:4]
        y_div = y_pred[:,4:7]
        y_inc = y_pred[:,7:]
        
        rho_mom = y_mom[:,0]
        rhou_mom = y_mom[:,1:3]
        p_mom = y_mom[:,-1]
        
        rho_div = y_div[:,0]
        rhou_div = y_div[:,1:3]
        
        # Get the derivatives
        Dy_pred = vmap(jacrev(self.forward_single))(x_pde)
        
        Dy_mom = Dy_pred[:,:4]
        Dy_div = Dy_pred[:,4:7]
        Dy_inc = Dy_pred[:,7:]
        
        Drho_mom = Dy_mom[:,0]
        Drhou_mom = Dy_mom[:,1:3]
        Dp_mom = Dy_mom[:,-1]
        
        Drho_div = Dy_div[:,0]
        Drhou_div = Dy_div[:,1:3]
        
        Du_mom = vmap(jacrev(self.forward_u_single))(x_pde)
        u_mom = rhou_div/rho_div.unsqueeze(1)
        
        if alignment_mode == 'DERL':
            alignment_loss = self.loss_container(Du_mom - Dy_inc, torch.zeros_like(Du_mom))
            alignment_loss+= self.loss_container(Dy_mom[:,:3] - Dy_div, torch.zeros_like(Dy_mom[:,:3]))
        elif alignment_mode == 'OUTL':
            alignment_loss = self.loss_container(u_mom - y_inc, torch.zeros_like(u_mom))
            alignment_loss+= self.loss_container(y_mom[:,:3] - y_div, torch.zeros_like(y_mom[:,:3]))
        elif alignment_mode == 'SOB':
            alignment_loss = self.loss_container(Du_mom - Dy_inc, torch.zeros_like(Du_mom)) + self.loss_container(u_mom - y_inc, torch.zeros_like(u_mom))
            alignment_loss+= self.loss_container(Dy_mom[:,:3] - Dy_div, torch.zeros_like(Dy_mom[:,:3])) + self.loss_container(y_mom[:,:3] - y_div, torch.zeros_like(y_mom[:,:3]))
        else:
            raise ValueError('alignment mode not recognized')
        return alignment_loss
