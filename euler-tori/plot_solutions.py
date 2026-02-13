import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata as int_grid
from mpl_toolkits.axes_grid1 import make_axes_locatable


#plotting function to generate the figures for the Tori problem
def plotVelDensTori(u,rho,T=[0,0.15,0.3],apx="",colorbar=True):
    num = len(T)
    fig,ax = plt.subplots(2,num,figsize=(21,14))
    
    plotVelsTori(u,T,ax[0])
    plotDensTori(rho,T,ax[1])
    
    fig.savefig("plots/2d_tori_{}.png".format(apx))
    plt.close(fig)

    
def plotVelsTori(u,T,ax1,ax2=None,rf_sols=None):
    N = 250
    a=1
    X,Y = np.meshgrid(np.linspace(0,a,N),np.linspace(0,a,N))
    
    for i,t in enumerate(T):
        pts = jnp.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1)]).T
        vel = vmap(u)(pts)
        U = np.array(vel[:,0].reshape(X.shape))
        V = np.array(vel[:,1].reshape(Y.shape))
        ax1[i].set_xlim(0,a)
        ax1[i].set_ylim(0,a)
        plt_str = ax1[i].streamplot(X,Y,U,V,density=0.45,arrowsize=4,linewidth=4,color='k')
        ax1[i].axis('off')
        
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
        ax1[i].add_patch(rect)
        
        if ax2 is not None:
            rf = rf_sols[i]
            points = np.array(rf.points[:,:2])
            rho_ref = np.array(rf['rho_n'])
            u_ref = np.array(rf['u_n'])

            rho_ref = int_grid(points,rho_ref,(X,Y))
            u_ref = int_grid(points,u_ref,(X,Y))

            ax2[i].set_xlim(0,a)
            ax2[i].set_ylim(0,a)
            #ax[0].plot(bd[0],bd[1])
            U_1 = np.array(u_ref[:,:,0])
            V_1 = np.array(u_ref[:,:,1])
            plt_str = ax2[i].streamplot(X,Y,U_1,V_1,density=0.45,arrowsize=4,linewidth=4,color='k')
            ax2[i].axis('off')

            rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k', facecolor='none')
            ax2[i].add_patch(rect)

def plotDensTori(rho,T,ax1,ax2=None,rf_sols=None,colorbar=True,clim=None):
    N = 250
    a=1
    X,Y = np.meshgrid(np.linspace(0,a,N),np.linspace(0,a,N))

    for i,t in enumerate(T):
        pts = jnp.vstack([np.ones(X.reshape(-1).shape)*t,X.reshape(-1),Y.reshape(-1)]).T
        dens = vmap(rho)(pts).reshape(X.shape)
        dens = np.array(dens)
        
        ax1[i].set_xlim(0,a)
        ax1[i].set_ylim(0,a)
        if clim:
            plt_dens1 = ax1[i].contourf(X,Y,dens,150,vmin=clim[0],vmax=clim[1])
        else:
            plt_dens1 = ax1[i].contourf(X,Y,dens,150)
        ax1[i].axis('off')
        
        if ax2 is not None:
            rf = rf_sols[i]
            points = np.array(rf.points[:,:2])
            rho_ref = np.array(rf['rho_n'])
            rho_ref = int_grid(points,rho_ref,(X,Y))

            ax2[i].set_xlim(0,a)
            ax2[i].set_ylim(0,a)
            #ax[0].plot(bd[0],bd[1])
            plt_dens2 = ax2[i].contourf(X,Y,rho_ref,150,vmax=clim[1],vmin=clim[0])
            ax2[i].axis('off')
    
    if colorbar:
        divider1 = make_axes_locatable(ax1[-1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.10)
        plt.colorbar(plt_dens1,cax=cax1)
        
        
