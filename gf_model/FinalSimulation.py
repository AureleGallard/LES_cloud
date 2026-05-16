'''
06/05/2025
Rebecca Gjini

Script is designed to run MCHammer for each LES simulation for the accretion KTF model.
The function created is made to make it easy to run each simulation on a super computer.
'''

# Import packages
import emcee
import numpy as np
import os
from pathlib import Path
#import from my scripts
import GFmodel as gf

def run_MCHammer(fold_num, file_num):
    file_num_str = str(file_num).zfill(2)

    # Set tile width you want to use
    tile_width = 8

    # Set experiment type 
    exp_num = '_exp03'

    LES_path = str(Path(__file__).resolve().parent.parent.parent / 'LES') + '/'

    #load mean cycle
    noisy_lc = np.loadtxt((LES_path + 'mean/%d_' + file_num_str + '_infall.txt') %fold_num , delimiter = ',')
    
    #smaller number of data 
    #makes matrix to account for the fact that the data are sampled every 2 minutes and 
    #the MCMC samples every 0.1 minutes
    ny = len(noisy_lc)
    nx = (len(noisy_lc) - 1)*20 + 1
    spacing = 20
    H = np.zeros((ny, nx))
    print(H.shape)
    for i in range(0, ny): 
        H[i, i*spacing] = 1

        #load covariance matrix
    R = np.loadtxt((LES_path + 'cov/%d_' +
                                         file_num_str + '_infall.txt') %fold_num, delimiter = ',')

    #Calculating R_sqrt using SVD
    [u,l,q] = np.linalg.svd(R)
    R_sqrt = u@(np.identity(len(noisy_lc))*np.sqrt(l))

    #number of walkers for the ensemble
    nwalkers = 20
    #dimensions of the parameter space
    ndim = 4
    #locate data cycle extrema
    noisy_extrem = (np.argmax(noisy_lc))*20 

    #load prior samples
    my_samples = np.loadtxt('files/prior/prior_samples'+exp_num+'.txt', delimiter = ',')

    #Pick 1000 random samples and find the nwalkers that give the highest probability values
    my_random_samples = np.random.randint(0, len(my_samples), 1000)

    my_log_probs = np.zeros(len(my_random_samples))
    for tt in range(0, len(my_log_probs)): 
        testing = my_samples[my_random_samples[tt]]
        my_log_probs[tt] = gf.log_post(testing, noisy_lc, noisy_extrem, nx, u, l, H)


    x_samples = my_samples[my_random_samples[np.argsort(my_log_probs)[-nwalkers:]]]
    #print(npr.log_post(x_samples[-1], noisy_lc, noisy_extrem, nx, u, l, H))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = ('files/mcmc_runs/%d_' +
            file_num_str + '_mcmc_infall' + exp_num + '.h5') %fold_num 
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    #Running the MC Hamer sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, gf.log_post, 
                                    args = [noisy_lc, noisy_extrem, nx, u, l, H],  backend=backend)

    sampler.run_mcmc(x_samples, 100) #10^5

    print('In the prior?', bool(gf.prior(sampler.get_chain()[-1,4])))

    #Actual Test
    test_length = 1000 #20
    sampler.reset()
    backend.reset(nwalkers, ndim)
    sampler.run_mcmc(x_samples, 100)
    for i in range(0, test_length - 1): 
        sampler.run_mcmc(None, 100)
        print('Iteration number',  i, end = "\r")