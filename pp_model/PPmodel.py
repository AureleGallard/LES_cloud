'''
This file contains the functions for the nonlinear cloud and rain equations with autoconversion
'''
import numpy as np
import scipy
from scipy.special import lambertw
import numba 
from numba import jit, njit

@njit
def dim_model(H_hist,H_eval,t,H_args):
    '''
    Calculates the delayed differential equations used to predict cloud cyles
    
    H_hist : List of floats array
        the history of the (past steps the have been taken)
    H_eval : float
        the value of the cloud depth H at time t
    t : float 
        time 
    H_args : List of floats array
        the arguments that will be passed into the model (H_0, tau, T, N, dt)
        
    Returns
    -------
    
    float : solution to differential equation at time t
    '''
    #define our input arguments 
    H0, tau, T , N, dt = H_args
    rho = (1e6/1440)/N
    if T == 0: HT=H_eval
    else: 
        past_time = int((t-T)/dt)
        HT = H_hist[past_time]
    #evaluate differential equation
    return((H0-H_eval)/tau - rho*HT**2)


@njit
def dim_RK4(H0, tau, T, N, Hint, solve_time, dt):
    '''
    This function is the Runge_Kutta4 differential equation solver for a delayed 
    differential equation that predicts cloud cycles
    
    H_0 : float
        cloud depth carrying capacity (in meters)
    tau : float
        characteristic time to reach carrying capacity (in minutes)
    T : float
        the delay associated with the time it takes to generate rain (in minutes)
    N : float
        droplet concentration (units are in m^(-3))
    Hint : List of floats array
        The initial values of Hint before time 0
    solve_time : List of floats array
        how long to simulate for
    dt : float
        time step
        
    Returns
    -------
    
    List of floats array: list of cloud depths 
    '''
    H_args = H0, tau, T , N, dt
    #How long the simulation needs to be with taking the delay into account
    odetime = np.arange(0,solve_time+T,dt)
    #Define solution array
    Y = np.zeros(len(odetime))
    #mumber of steps taken in the past
    past_steps = len(np.arange(0,T+dt,dt))
    #initialize Y values for the time delay (before time 0)
    for kk , time in enumerate(np.arange(0,T + dt,dt)):
        Y[kk] = Hint[kk-past_steps] 
    #use RK4 to calculate the H values for times past 0
    for jj , t_n in enumerate(odetime):
        if t_n < T+dt: pass
        else:
            y_n = Y[jj-1]
            k1 = dim_model(Y,y_n,t_n,H_args)
            y_eval = y_n + dt*k1/2.0
            k2 = dim_model(Y,y_eval,t_n+dt/2,H_args)
            y_eval = y_n + dt*k2/2.0
            k3 = dim_model(Y,y_eval,t_n+dt/2,H_args)
            y_eval = y_n + dt*k3
            k4 = dim_model(Y,y_eval,t_n+dt,H_args)
            Y[jj]=y_n+dt*(k1+2*k2+2*k3+k4)/6.0
    return(Y)

@njit 
def findLastMin(H): 
    '''
    Solves for the last minimum in a cloud cycle simulation
    
    H : List of floats array
        Cloud depth values from a cloud cycle simulation
        
    Returns
    -------
    float: index of the last minimum in the cloud depth array
    '''
    #finds all the local minimums
    extrema_index = np.where((H[1:-1] < H[0:-2]) * (H[1:-1] < H[2:]))[0] + 1 
    #Check if there are no local minimums
    if not np.any(extrema_index):
        return 0
    else: 
        #returns the last minimum
        return extrema_index[-1]
@njit
def FindNextMin(H, lastMin): 
    '''
    Solves for the second to last minimum in a cloud cycle simulation that comes before the 
    actual last minimum
    
    H : List of floats array
        Cloud depth values from a cloud cycle simulation    
    lastMin : float
        index of the last minimum in the cloud cycle simulation
        
    Returns
    -------
    
    float : index of the next last minimum in the cloud depth array
    '''
    #while loop stop criteria
    go = 1
    #find the last minimum of a shortened cloud cycle array
    nextMin = findLastMin(H[0:lastMin])
    #check whether the minimum is close to the value of the last minimum
    while (go == 1): 
        if (np.abs(H[lastMin]-H[nextMin])/np.abs(H[lastMin]) < 0.1): 
            go = 0
        else: 
            nextMin = findLastMin(H[1:nextMin])
            if nextMin == 0:
                go = 0
    return nextMin

@njit
def FindLimitCycle(H_sim, theta, solve_time = 1440, dt= 0.1): 
    '''
    Solves for the limit cycle of a cloud cycle simulation
    
    H_sim : List of floats array
        cloud depth from the cloud cycle simulation    
    theta : List of floats array
        parameters that created the cloud cycle simulation
    solve_time : float 
        amount of time to simulate for and to continue to simulate for (default is one day in minutes)
    dt : float
        time step
        
    Returns
    -------
    
    List of floats array: list of cloud depths 
    
    '''
    H = H_sim
    H0 = theta[0]
    tau = theta[1]
    T = theta[2]
    N = theta[3]
    
    for i in range(0, 11): 
        for j in range(0, len(H)): 
            if (H[j] > 1.0e6): 
                return (-1)*np.ones(10)
        lastMin = findLastMin(H)
        nextMin = FindNextMin(H,lastMin)
        if nextMin == 0:
            Hint = H[-len(np.arange(0,T+dt,dt)): -1]
            H2 = dim_RK4(H0, tau, T, N, Hint, solve_time, dt)
            H = np.concatenate((H, H2[len(Hint) - 1:]))
        else:
            C1 = [nextMin, lastMin]
            HC1 = H[C1[0]:C1[1]]
            nextMin = FindNextMin(H,nextMin)
            if nextMin == 0:
                Hint = H[-len(np.arange(0,T+dt,dt)): -1]
                H2 = dim_RK4(H0, tau, T, N, Hint, solve_time, dt)
                H = np.concatenate((H, H2[len(Hint) - 1:]))
            else:
                C2 = [nextMin,C1[0]]
                HC2 = H[C2[0]:C2[1]]
                C1L=C1[1]-C1[0]
                C2L=C2[1]-C2[0]
                if C2L>C1L:
                    
                    HC2 = H[C2[0]:C2[0]+C1L] 
                else:
                    
                    HC1 = H[C1[0]:C2L + C1[0]]
                    C1 = [C1[0], C1[0] + C2L]
                rmse = np.sqrt(np.sum((HC1-HC2)**2)/len(HC1))
                if rmse < 10:
                    LCycle = HC1 
                    return LCycle 
                else:
                    Hint = H[-len(np.arange(0,T+dt,dt)): -1]
                    H2 = dim_RK4(H0, tau, T, N, Hint, solve_time, dt)
                    H = np.concatenate((H, H2[len(Hint) - 1:]))
                        
    return (-1)*np.ones(10)

#prior probability function 
def prior(theta, max_H0 = 4000, max_tau = 720, max_T = 720, 
          max_N = 1e8): 
    '''
    Solves for prior probability of a set of cloud parameters theta
       
    theta : List of floats array
        parameters that created the cloud cycle simulation
  
    Returns
    -------
    
    float : either 1 or 0, 0 for 0 probability (unsatisfied) 
    and 1 for probabilty 1 (satisfied conditions)
    
    '''
    H_0 = theta[0]
    tau = theta[1]
    T = theta[2]
    N = theta[3]
    dt = 0.1
    solve_time = 1440
    Hint = 0.1*np.ones(len(np.arange(0,T+dt,dt)))
    
    if any(th < 0 for th in theta[0:4]):
        return 0
    
    if (H_0 > max_H0 or tau > max_tau or T > max_T or N > max_N): 
        return 0
    
    if T >= tau : 
        return 0
    
    mu = N/((1e6/1440)*tau*H_0)
    in_w = (-2)*(np.sqrt((1/mu) + 0.25) - 0.5)*(T/tau)*np.exp(T/tau)
    beta = (tau/T)*lambertw(in_w) - 1
    
    if beta.real <= 0: 
        return 0

    model_run = dim_RK4(H_0, tau, T, N, Hint, solve_time, dt) 
    

    myLimitCycle = FindLimitCycle(model_run, theta) 
    
    if (any(m < 0 for m in myLimitCycle) or not np.any(scipy.signal.argrelextrema(myLimitCycle, np.greater)[0])):
        return 0
    else:
        return 1
    
#posterior function using SVD
def log_post(theta, Hd, Hd_max, og_len, u, l, H_in): 
    '''
    Solves for log posterior probability of a set of cloud 
    parameters theta against a cloud cycle data set
       
    theta : List of floats array
        parameters that created the cloud cycle simulation
    Hd : List of floats array
        cloud cycle data
    Hd_max : integer
        argmax of cloud cycle data
    og_len : float
        length of the cycle data (minutes*10 -> samples are taken every 0.1 minutes)
    u : 2D float array
        U matrix from an SVD factorization of the covariance matrix of the data
    l : List of floats array
        eigen value list from a SVD factorization
    H_in : 2D float array
        matrix of 1's and 0's that parses the simulated output to match
        the data array size 
  
    Returns
    -------
    
    float : log value of the posterior probability of a set of cycle parameters theat
    
    '''
    test_prior = prior(theta)
    if (test_prior == 0): 
        return (-1)*np.inf
    else: 
        H0 = theta[0]
        tau = theta[1]
        T = theta[2]
        N = theta[3]
        
        dt = 0.1 
        solve = 1440
        Hint = 0.1*np.ones((len(np.arange(0, T + dt, dt))))

        H = dim_RK4(H0, tau, T, N, Hint, solve, dt)
        M = FindLimitCycle(H, theta)
        lHd = og_len
        lM = len(M)
        ############################################################
        M_max = np.argmax(M)
        if(lM - M_max > lHd - Hd_max): 
            M = M[:M_max + lHd - Hd_max]
        else: 
            M = np.concatenate((M, np.zeros((lHd - Hd_max )- (lM -M_max))))
        
        if(M_max > Hd_max): 
            M = M[M_max - Hd_max:]
        else: 
            M = np.concatenate((np.zeros(Hd_max - M_max), M))
        return       -0.5*(np.linalg.norm((u.T@(Hd - H_in@M))/np.sqrt(l))**2)
    
def find_spinup_end(cloud_fraction, tol = 0.001): 
    cfs = cloud_fraction[100:]
    start_ind = 0
    kk = 0 
    for i in range(0,260): 
        kk = kk + 1
        cfs_old = cfs 
        cfs = scipy.signal.detrend(cfs)
        cfs = cfs - np.mean(cfs)
        ind = np.argwhere(cfs < 0)[0][0]
        cfs = cfs[ind:]
        start_ind = start_ind + ind
        if (kk > 1): 
            n_new = len(cfs)
            n_old = len(cfs_old)
            cf_new = np.zeros(n_old)
            cf_new[n_old - n_new:] = cfs
            if np.linalg.norm(cf_new-cfs_old)<tol: 
                return start_ind + 100