#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_parameters import * 
from mcmc import *
import multiprocessing           
from multiprocessing import Pool, set_start_method  
import emcee                   
import os

####################################################################################
# This module starts the MCMC run, building on the parameter definitions in check global_parameters.py
# It first checks if the file to store the results exists
# and if so, it deletes and creates a new file (check global_parameters.py for filename)
# If vectorise = True, the code inputs the data into the mcmc engine as a matrix (Nwalkers x Ndim) and parallelises 
# on GPU. 
# If vectorise = False, the code parallelises the walkers through the multi-processing framework. 
####################################################################################

if os.path.isfile(filename)==True: 
    os.remove(filename)
    print(color.RED + 'previous MCMC data file removed' + color.END)
    print(color.BOLD + 'data saved at: ' + filename + color.END)
    print(color.BLACK +'starting MCMC run .. '+  color.END)

backend = emcee.backends.HDFBackend(filename)
backend.reset(Nwalker, Ndim)

if vectorize == True:
    sampler = emcee.EnsembleSampler(Nwalker, Ndim, lnprob_vec, backend=backend, vectorize = True)
    pos, prob, state = sampler.run_mcmc(p_init_MC, Nsteps, progress= True)
        
else: 
    multiprocessing.set_start_method('spawn', force=True)
    if __name__ == "__main__":
        multiprocessing.set_start_method('spawn', force=True)
        with Pool(processes = Nwalker) as pool:
            sampler = emcee.EnsembleSampler(Nwalker, Ndim, lnprob, backend=backend, pool = pool, vectorize = False)
            pos, prob, state = sampler.run_mcmc(p_init_MC, Nsteps, progress= True)


##### Use these for not-parallelised walkers. This makes the MCMC run slower.
#sampler = emcee.EnsembleSampler(Nwalker, Ndim, lnp\rob, backend=backend)
#pos, prob, state = sampler.run_mcmc(p_init_MC, Nsteps, progress=True)
