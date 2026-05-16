"""
Cloud plotter
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d as gf1d

def Colors(plot_pallete = False,return_pallet= False):
    if plot_pallete:
        sns.palplot(sns.color_palette("Paired",12))
        plt.show()
    scp = sns.color_palette("Paired",12)
    if return_pallet: 
        return(scp)
    Colors = {'LES': scp[1],
             'LES_err': scp[0],
             'KTF17': scp[3],
             'KTF17_err': scp[2]}
    return(Colors)

def Labels(print_labels= False):
    labels = ['$H_0$, km','$\\tau$, min','$T$, min','$\\alpha$, d${}^{-1}$m${}^{-2.5}$']
    if print_labels:
        print(labels)
    return(labels)

def Bounds(print_bounds= False):
    bounds=[[.5,4],[10,0.18*1440],[10,0.05*1440],[100,1000]]
    if print_bounds:
        print(bounds)
    return(bounds)
        

def change_units(samples):
    shape = samples.shape
    if len(shape)==2:
        # H0 to km
        samples[:,0] /= 1000   
        # tau to min
        samples[:,1] *= 1440
        # T to min
        samples[:,2] *= 1440
    if len(shape)==3:
        # H0 to km
        samples[:,:,0] /= 1000
        # tau to min
        samples[:,:,1] *= 1440
        # T to min
        samples[:,:,2] *= 1440
    return(samples)

def unchange_units(samples):
    shape = samples.shape
    if len(shape)==2:
        # H0 to m
        samples[:,0] *= 1000
        # tau to min
        samples[:,1] /= 1440
        # T to min
        samples[:,2] /= 1440
    if len(shape)==3:
        # H0 to m
        samples[:,:,0] *= 1000
        # tau to min
        samples[:,:,1] /= 1440
        # T to min
        samples[:,:,2] /= 1440
    return(samples)

def dual_triangle(chain,lnprob,
                  cmap_scatter=cm.summer,cmap_hist=cm.Blues,size = 0.05,alpha = 0.3,linewidth= 10,ndx=10):
    labels = Labels()
    bounds = Bounds()

    d1,d2,nDim = chain.shape
    data = chain.reshape(d1*d2,nDim)
    nSamps , nDim = data.shape
    nBins = 1000
    param_range = []
    X_plot = []

    fig, ax = plt.subplots(nrows=nDim, ncols=nDim, figsize=(nDim**2, nDim**2))
    for kk in range(0,nDim):
        for jj in range(kk+1,nDim):
            ax[kk,jj].axis('off')

        ax[kk,kk].set_xlim(bounds[kk])
        ax[kk,kk].set_yticks([])
        ax2 = ax[kk,kk].twinx()
        ax2.hist(data[:,kk], bins = 50, histtype=u'step', density=False, stacked=False,linewidth = 7,color='k') #linewidth
        ax2.set_yticks([])
        if kk != nDim-1:
            ax[kk,kk].set_xticks([])

        for jj in range(kk):
            max_P = np.exp(max_in_box(data[:,jj],data[:,kk],lnprob.reshape(d1*d2),bounds[jj],bounds[kk],ndx))
            ax[kk,jj].imshow(max_P,cmap =cmap_scatter ,aspect = 'auto',extent=(bounds[jj][0],bounds[jj][1],bounds[kk][0],bounds[kk][1]))
            counts,xbins,ybins = np.histogram2d(data[:,jj],data[:,kk],bins=20) 
            ax[kk,jj].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=linewidth,cmap=cmap_hist)
            ax[kk,jj].set_xlim(bounds[jj])
            ax[kk,jj].set_ylim(bounds[kk])
            if jj != 0:
                ax[kk,jj].set_yticks([])
            if kk != nDim-1:
                ax[kk,jj].set_xticks([])

    for jj in range(4):
        ax[3,jj].set_xlabel(labels[jj])
    for jj in range(1,4):
        ax[jj,0].set_ylabel(labels[jj])
    plt.show()
'''   
def get_trajectories_v2a(samples,cloud_H,nSamps):
    nChain , nDim = samples.shape

    Y = np.zeros((len(cloud_H),nSamps))
    for kk in range(nSamps):
        ii = np.random.randint(nChain)
        H0,tau,T,alpha = samples[ii,:]
        Y[:,kk]= Centered_Mean_v2a.MCMC_model(H0,tau,T,alpha,cloud_H)
    return(Y)
'''

def get_cycle_stats(Y):
    _,nSamps = Y.shape
    stats = np.zeros((5,nSamps)) # period, amplitude, growth_time, decay_time, max_height
    for kk in range(nSamps):
        tmid = np.argmax(Y[:,kk])
        tmp = Y[:,kk]>0
        tmp = np.where(tmp == 1)[0]
        t0 , t1 = tmp[0],tmp[-1]

        stats[0,kk] = t1-t0+1
        stats[1,kk] = Y[tmid,kk] - Y[t0,kk]
        stats[2,kk] = tmid - t0+1
        stats[3,kk] = t1 - tmid +1
        stats[4,kk] = Y[tmid,kk]
    return(stats)


def contours(chain,cmap_hist=cm.autumn,linewidth= 7,fill_contours = True,nLevels = 25,bounds = None,nBins = 20,
             labels = None):
    if labels == None:
        labels = Labels()
    if bounds == None:
        bounds = Bounds()
    
    d1,d2,nDim = chain.shape

    data = chain.reshape(d1*d2,nDim)
    nSamps , nDim = data.shape

    fig, ax = plt.subplots(nrows=nDim, ncols=nDim, figsize=(nDim**2, nDim**2))
    for kk in range(0,nDim):
        for jj in range(kk+1,nDim):
            ax[kk,jj].axis('off')

        ax[kk,kk].set_xlim(bounds[kk])
#        ax[kk,kk].hist(data[:,kk],histtype=u'step', normed=True,linewidth = linewidth,color='k',bins=int(5*nBins))
        
###        
        counts,xbins = np.histogram(data[:,kk], density=True,bins=int(2*nBins))
        xx = [xbins[0]]+[(xbins[mm]+xbins[mm+1])/2 for mm in range(int(2*nBins))]+[xbins[-1]]
        yy = [0] + list(counts)+[0]
        yy = gf1d(yy,2)
        ax[kk,kk].plot(xx,yy,linewidth = linewidth,color='k')
        
        
###        
        
        ax[kk,kk].set_yticks([])
        if kk != nDim-1:
            ax[kk,kk].set_xticks([])

        for jj in range(kk):
            counts,xbins,ybins = np.histogram2d(data[:,jj],data[:,kk],bins=nBins,range=[bounds[jj],bounds[kk]],density=True)
            Levels = np.linspace(0,np.max(counts),nLevels+1)
            if fill_contours:
                ax[kk,jj].contourf(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],cmap=cmap_hist,levels =Levels )
            else:
                ax[kk,jj].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=linewidth,cmap=cmap_hist,levels =Levels)
            ax[kk,jj].set_xlim(bounds[jj])
            ax[kk,jj].set_ylim(bounds[kk])
            if jj != 0:
                ax[kk,jj].set_yticks([])
            if kk != nDim-1:
                ax[kk,jj].set_xticks([])
    for jj in range(4):
        ax[3,jj].set_xlabel(labels[jj])
    for jj in range(4):
        ax[jj,0].set_ylabel(labels[jj])
    
    for jj in range(4):
        ax[jj, 0].yaxis.set_label_coords(-0.5,0.5)
    return(fig,ax)
    
def max_in_box(X,Y,P,x_bounds,y_bounds,ndx):
    dx = (x_bounds[1]-x_bounds[0])/(ndx)
    dy = (y_bounds[1]-y_bounds[0])/(ndx)
    max_P = np.zeros((ndx,ndx))
    for jj in range(ndx):
        if jj == 0: yI1 = Y >= y_bounds[0]+jj*dy
        else: yI1 = Y > y_bounds[0]+jj*dy
        yI2 = Y <= y_bounds[0]+(jj+1)*dy

        for kk in range(ndx):
            if kk == 0: xI1 = X >= x_bounds[0]+kk*dx
            else: xI1 = X > x_bounds[0]+kk*dx
            xI2 = X <= x_bounds[0]+(kk+1)*dx


            Index = xI1*xI2*yI1*yI2
            if np.all(Index== False):
                max_P[jj,kk] = -np.inf
            else:
                max_P[jj,kk] = np.max(P[Index])
    return(max_P[::-1])

def LES_err_vs_KTF17_err(cloud_H,Cov,Y,std = 2, alpha = 0.5):
    
    # Y is an array of KTF17 trajectories
    
    nPts , nSamps = Y.shape
    colors = Colors()

    err = std*(np.diag(Cov)-1E4)**0.5
    yerr = np.zeros((2,len(err)))
    yerr[1,:]=err

    yerr[0,np.where(cloud_H-err<0)]=cloud_H[np.where(cloud_H-err<0)]
    yerr[0,np.where(cloud_H-err>0)]=err[np.where(cloud_H-err>0)]

    x_plt = np.linspace(0,nPts,nPts)
    plt.fill_between(x_plt,cloud_H-yerr[0,:],cloud_H+yerr[1,:],color=colors['LES_err'])

    for kk in range(nSamps):
        index = np.where(Y[:,kk]>0)[0]
        plt.plot(x_plt[index],Y[:,kk][index],linewidth=0.5,color =colors['KTF17'],alpha = alpha)

    plt.plot(cloud_H,linewidth = 10,color=colors['LES'])
    plt.ylabel('Cloud Depth, m')
    plt.xlabel('Time, min')


def dual_contours(chain1,chain2,cmap_hist=cm.autumn,linewidth= 7,fill_contours = True,nLevels = 25,bounds = None,nBins=20):
    labels = Labels()
    if bounds == None:
        bounds = Bounds()
    
    d1,d2,nDim = chain1.shape
    data1 = chain1.reshape(d1*d2,nDim)
    
    d1,d2,nDim = chain2.shape
    data2 = chain2.reshape(d1*d2,nDim)
    
    nSamps , nDim = data2.shape

    fig, ax = plt.subplots(nrows=nDim, ncols=nDim, figsize=(nDim**2, nDim**2))

    for kk in range(0,nDim):
        for jj in range(kk+1,nDim):
#            ax[kk,jj].axis('off')
            ax[kk,jj].tick_params(colors='b')

        ax[kk,kk].set_xlim(bounds[kk])
#        ax[kk,kk].hist(data1[:,kk],histtype=u'step', normed=True,linewidth = linewidth,color='k',bins=int(5*nBins))
        
###        
        counts,xbins = np.histogram(data1[:,kk], density=True,bins=int(2*nBins))
        xx = [(xbins[mm]+xbins[mm+1])/2 for mm in range(int(2*nBins))]
        counts = gf1d(counts,2)
        ax[kk,kk].plot(xx,counts,linewidth = linewidth,color='k')
### 
        
#        ax[kk,kk].hist(data2[:,kk],histtype=u'step', normed=True,linewidth = linewidth,color='b',bins=int(5*nBins))

###        
        counts,xbins = np.histogram(data2[:,kk], normed=True,bins=int(2*nBins))
        counts = gf1d(counts,2)
        xx = [(xbins[mm]+xbins[mm+1])/2 for mm in range(int(2*nBins))]
        ax[kk,kk].plot(xx,counts,linewidth = linewidth,color='b')
### 
        ax[kk,kk].set_yticks([])
        if kk != nDim-1:
            ax[kk,kk].set_xticks([])

        for jj in range(kk):
            counts1,xbins1,ybins1 = np.histogram2d(data1[:,jj],data1[:,kk],bins=nBins,range=[bounds[jj],bounds[kk]],normed = True)
            Levels = np.linspace(0,np.max(counts1),nLevels+1)
            if fill_contours:
                ax[kk,jj].contourf(counts1.T,extent=[xbins1.min(),xbins1.max(),ybins1.min(),ybins1.max()],cmap=cmap_hist,levels =Levels )
            else:
                ax[kk,jj].contour(counts1.T,extent=[xbins1.min(),xbins1.max(),ybins1.min(),ybins1.max()],linewidths=linewidth,cmap=cmap_hist,levels =Levels)
            ax[kk,jj].set_xlim(bounds[jj])
            ax[kk,jj].set_ylim(bounds[kk])
            if jj != 0:
                ax[kk,jj].set_yticks([])
            if kk != nDim-1:
                ax[kk,jj].set_xticks([])
            
            counts2,xbins2,ybins2 = np.histogram2d(data2[:,kk],data2[:,jj],bins=nBins,range=[bounds[kk],bounds[jj]],normed = True)
            Levels = np.linspace(0,np.max(counts2),nLevels+1)
            if fill_contours:
                ax[jj,kk].contourf(counts2.T,extent=[xbins2.min(),xbins2.max(),ybins2.min(),ybins2.max()],cmap=cmap_hist,levels =Levels )
            else:
                ax[jj,kk].contour(counts2.T,extent=[xbins2.min(),xbins2.max(),ybins2.min(),ybins2.max()],linewidths=linewidth,cmap=cmap_hist,levels =Levels)
            ax[jj,kk].set_xlim(bounds[kk])
            ax[jj,kk].set_ylim(bounds[jj])
            if kk != nDim-1:
                if kk != 0:
                    ax[jj,kk].set_yticks([])
            if jj !=0:
                if jj != nDim-1:
                    ax[jj,kk].set_xticks([])
    
    
    for jj in range(4):
        ax[3,jj].set_xlabel(labels[jj])
        ax[0,jj].xaxis.set_ticks_position('top')
    for jj in range(4):
        ax[jj,0].set_ylabel(labels[jj])
        ax[jj,3].yaxis.tick_right()
        
    for jj in range(4):
        ax[jj, 0].yaxis.set_label_coords(-0.5,0.5)
        
        
def dual_1dmarg(chain1,chain2,cmap_hist=cm.autumn,linewidth= 7,fill_contours = True,nLevels = 25,bounds = None,nBins=20,marg1d_color = ['k','b']):
    labels = Labels()
    if bounds == None:
        bounds = Bounds()
    
    d1,d2,nDim = chain1.shape
    data1 = chain1.reshape(d1*d2,nDim)
    
    d1,d2,nDim = chain2.shape
    data2 = chain2.reshape(d1*d2,nDim)
    
    nSamps , nDim = data2.shape

    fig, ax = plt.subplots(nrows=nDim, ncols=nDim, figsize=(nDim**2, nDim**2))

    for kk in range(0,nDim):
        for jj in range(kk+1,nDim):
            ax[kk,jj].axis('off')
        ax[kk,kk].set_xlim(bounds[kk])
#        ax[kk,kk].hist(data2[:,kk],histtype=u'step', normed=True,linewidth = linewidth,color=marg1d_color[1],bins=int(5*nBins))
        
###        
        counts,xbins = np.histogram(data2[:,kk], density=True,bins=int(2*nBins))
        xx = [xbins[0]]+[(xbins[mm]+xbins[mm+1])/2 for mm in range(int(2*nBins))]+[xbins[-1]]
        yy = [0] + list(counts)+[0]
        yy = gf1d(yy,2)
        ax[kk,kk].plot(xx,yy,linewidth = linewidth,color=marg1d_color[1])
### 
        
#        ax[kk,kk].hist(data1[:,kk],histtype=u'step', normed=True,linewidth = linewidth,color=marg1d_color[0],bins=int(5*nBins))
###        
        counts,xbins = np.histogram(data1[:,kk], density=True,bins=int(2*nBins))
        xx = [xbins[0]]+[(xbins[mm]+xbins[mm+1])/2 for mm in range(int(2*nBins))]+[xbins[-1]]
        yy = [0] + list(counts)+[0]
        yy = gf1d(yy,2)
        ax[kk,kk].plot(xx,yy,linewidth = linewidth,color=marg1d_color[0])
### 

        ax[kk,kk].set_yticks([])
        if kk != nDim-1:
            ax[kk,kk].set_xticks([])
            
        for jj in range(kk):
            counts,xbins,ybins = np.histogram2d(data1[:,jj],data1[:,kk],bins=nBins,range=[bounds[jj],bounds[kk]],density = True)
            Levels = np.linspace(0,np.max(counts),nLevels+1)
            if fill_contours:
                ax[kk,jj].contourf(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],cmap=cmap_hist,levels =Levels )
            else:
                ax[kk,jj].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=linewidth,cmap=cmap_hist,levels =Levels)
            ax[kk,jj].set_xlim(bounds[jj])
            ax[kk,jj].set_ylim(bounds[kk])
            if jj != 0:
                ax[kk,jj].set_yticks([])
            if kk != nDim-1:
                ax[kk,jj].set_xticks([])
    for jj in range(4):
        ax[3,jj].set_xlabel(labels[jj])
    for jj in range(4):
        ax[jj,0].set_ylabel(labels[jj])
    
    for jj in range(4):
        ax[jj, 0].yaxis.set_label_coords(-0.5,0.5)



def my_tri_plotter(chain,lnprob,
                  cmap_scatter=cm.summer, cmap_hist=cm.Blues,linewidth_r = 10,ndx=10, linewidth_h = 7,
                  nLevels = 25, bounds = None, nBins = 50, ncBins = 50):


    labels = Labels()
    if bounds == None:
        bounds = Bounds()

    d1,d2,nDim = chain.shape
    data = chain.reshape(d1*d2,nDim)
    nSamps , nDim = data.shape


    fig, ax = plt.subplots(nrows=nDim, ncols=nDim, figsize=(nDim**2, nDim**2))
    for kk in range(0,nDim):
        for jj in range(kk+1,nDim):
            ax[kk,jj].axis('off')


        ax[kk,kk].set_xlim(bounds[kk])

        counts,xbins = np.histogram(data[:,kk], density=True,bins=int(2*nBins))
        xx = [xbins[0]]+[(xbins[mm]+xbins[mm+1])/2 for mm in range(int(2*nBins))]+[xbins[-1]]
        yy = [0] + list(counts)+[0]
        yy = gf1d(yy,2)
        ax[kk,kk].plot(xx,yy,linewidth = linewidth_h,color='k')
        
        ax[kk,kk].set_yticks([])
        if kk != nDim-1:
            ax[kk,kk].set_xticks([])


        for jj in range(kk):
            max_P = np.exp(max_in_box(data[:,jj],data[:,kk],lnprob.reshape(d1*d2),bounds[jj],bounds[kk],ndx))
            ax[kk,jj].imshow(max_P,cmap =cmap_scatter ,aspect = 'auto',extent=(bounds[jj][0],bounds[jj][1],bounds[kk][0],bounds[kk][1]))
            counts,xbins,ybins = np.histogram2d(data[:,jj],data[:,kk],bins=ncBins) 
            ax[kk,jj].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                                linewidths=linewidth_r,cmap=cmap_hist)
            ax[kk,jj].set_xlim(bounds[jj])
            ax[kk,jj].set_ylim(bounds[kk])
            if jj != 0:
                ax[kk,jj].set_yticks([])
            if kk != nDim-1:
                ax[kk,jj].set_xticks([])

    for jj in range(4):
        ax[3,jj].set_xlabel(labels[jj])
    for jj in range(1,4):
        ax[jj,0].set_ylabel(labels[jj])
    plt.show()