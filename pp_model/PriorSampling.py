'''
Prior sampling script for super computer to run

By: Rebecca Gjini
'''
import numpy as np
import PPmodel as pp

file_path = "files/prior/"
def test_fun(num_s):
    samples = np.zeros((num_s,4))
    count = 0
    for i in range(0, num_s): 
        uniform = np.zeros(4)
        uniform[0] = np.random.uniform(0, 4000) #H_0
        uniform[1] = np.random.uniform(0, 720) #tau in minutes
        uniform[2] = np.random.uniform(0, 720) #T 
        uniform[3] = np.random.uniform(0, 1e8) #N
        if pp.prior(uniform) == 1: 
            samples[count] = uniform
            count = count + 1
            # print("The number of samples is:", count, 'and total iterations is', i, end = "\r")
    return samples[:count]

num_try = 500000
for jj in range(0, 10): 
    np.random.seed(jj)
    my_new_samples = test_fun(num_try)
    np.savetxt((file_path + "prior_samples_%d.txt") %jj, my_new_samples, delimiter = ',') 