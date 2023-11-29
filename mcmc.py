#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_parameters import *  # import parameter values 
from waveform import *           # import kludge equations and ODE solvers
from propagation import *        # import GW propagation functions
import numpy as np
import cupy as cp 
import emcee                     # MC package
import multiprocessing           # parallelisation 
from multiprocessing import Pool # parallelisation 


# Definition of the prior. We choose a flat prior.
# Change lnprior(p) accordingly for different set/number of parameters.
def lnprior(p):
    """
    Defines the log-prior.

Parameters
----------
p : 1D array, parameter values, p = [M, μ, spin, ...]

Returns
-------
real, log(prior)
    """  
    x, y, z, w = p 
# x: mass M, y: mass μ, z: spin S, w: Xi. 'Xi' is an example modified gravity parameter we use 
# (see 'propagation.py' for its definition). 
    if ( 
                    10**5  <= x <=   10**7 
        and             1  <= y <=   100
        and             0  <= z <=   1.0
        and          -100  <= w <= 100
        ):              
        return 0   
    else:
                return -np.inf
########### 


# Definition of the log-likelihood. By construction, it is infinite outside the margins of our parameter space.
# It needs to be modified for different set of parameters.
def lnprob(p):
    """
    Defines the log-posterior probability function for the MCMC run. 

Parameters
----------
p : 1D array, parameter values, p = [M, μ, spin, ...]

Returns
-------
real, log(prior) + log(likelihood) 
    """   
    x, y, z, w     = p                            # Change this vector of parameters if needed.
    fiducial_model = compute_fiducial(x0,p0)      # This is needed to evaluate the function iterate_mcmc()
    #print('[', p[0],',',p[1],',',p[2], ',' ,']') # Prints out the chosen parameters at each MCMC step.
    if not np.isfinite(lnprior(p)):               # Check prior's boundaries 
        return -np.inf
    return lnprior(p) + iterate_mcmc(x0, p, fiducial_model) # returns log(prob) = ln(prior) + ln(likelihood)
################## 



# This is the detector noise as in "noise(f)", re-written for GPU/cuda parallelisation. It is used in the MCMC. It # takes as input a frequency and returns the noise value for LISA. See Barrack&Cutler2004. For a more recent 
# expression for the noise function see below.
get_noise = cp.ElementwiseKernel(
        'T f',
        'T S',
        '''
           T S_inst = (1.22*pow(10.0,-51.0))*pow(f,-4.0) + (1.22*pow(10.0,-37.0))*pow(f,2.0) + 2.12*pow(10.0,-41.0);
           T S_exgal = 4.2 * pow(10.0,-47.0) * pow(f,(-7.0/3.0));
           T S_gal = 2.1*pow(10.0,-45.0)*pow(f,(-7.0/3.0));
           S = S_inst + S_exgal + S_gal;
        ''',
        'get_noise'
    )


# This function takes as input the FFT values of 2 waveforms, together with a noise function, and returns the log-likelihood as: log-likelihood = (-0.5)*(waveform1 - waveform2)^2/noise. 
# It is re-written for GPU parallelisation and it is used in the MCMC.
get_Likelihood = cp.ElementwiseKernel(
        'T FFT0_gpu_1_j, T FFT_i_gpu_1_j, T FFT_i_gpu_0_j_noised',
        'T L_j',
        '''
            L_j = (-0.5) * pow((FFT0_gpu_1_j - FFT_i_gpu_1_j) , 2.0 )/FFT_i_gpu_0_j_noised;
        ''',
        'get_Likelihood'
    )



## iterate_mcmc()
def iterate_mcmc(x, p, fiducial_model):
    """
    Computes the waveform and its FFT for each step of the MCMC run, and returns the log-likelihood
    around the fiducial model weighted by the noise function. It is evaluated at each step of the 
    MCMC run through the function lnprob() below.

Parameters
----------
x : 1D array, initial conditions for the ODE solver. 
p : 1D array, parameter values at each step of the MCMC run. They are provided by the MCMC algorithm.
fiducial_model: 2D array, fiducial model as produced by the function compute_fiducial().
Returns
-------
2D array, [amplitude of waveform, FFT of amplitude of waveform] 
    """   
    np.seterr(divide='ignore')
    
    ###########  Parameters to be varied/fit in the MCMC. 
    # Must be of the same type with the ones in the fiducial model. Change them if needed. 
    M    = p[0]             # Central black hole mass 
    μ    = p[1]             # Orbiting star mass
    spin = p[2]             # Spin of central black hole
    Xi   = p[3]             # GW luminosoty distance modification compared to EM luminosity distance. 
    args = [p[0],p[1],p[2]] # This is passed as argument into the ODE solver below. 
                            # GW propagation/luminosity distance parameters DON'T ENTER here, but in the LIKELIHOOD computation .
     
    ### Initial conditions at LSO
    # x0 is the vector with initial conditions. In this example, all quantities are fixed and not varied/fit in the MCMC.
    x      = x0
    nu_LSO = (c**3/(2*Pi*G*M*Ms))*((1 - e_LSO**2)/(6 + 2*e_LSO))**(3/2) 
    # Notice that at each MC step the initial condition for nu_LSO needs to be updated since it depends on the central mass M.
    ###########


    # Solves the initial value problem with initial conditions = x0 and parameters = args.
    #sol_i = solve_ivp(eqs, (t_max, t_min), x0, method=solver0, t_eval = t_span, args=args, rtol=rtol, atol=atol)
    sol_i = compute_orbit(args, x0)

    #if sol_i.success == False: 
    if sol_i == 0: # Check that solution of ODEs exists without numerical issues.
        print("NO_SOLUTION")
        return -np.inf # If solution of ODEs does not exist, skip and give a zero probability in the MCMC run to  
                       # keep the MCMC going.
   
    else:
        #waveform_i = waveform(n_max, M0, μ0, spin0, sol_i.t, sol_i.y)  # Compute the waveform
        waveform_i = waveform(n_max, M0, μ0, spin0, sol_i[0],sol_i[1])  # Compute the waveform
        FFT_i_gpu  = FFT_gpu(waveform_i[0], waveform_i[1])              # Compute the FFT of the waveform           
        FFT0_gpu   = fiducial_model[1]                                  # The FFT-values of the fiducial modes        
        FFT_i_gpu_0_j_noised = get_noise(FFT_i_gpu[0])                  # The LISA noise response evaluated on the frequency grid  
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #  L_gpu_vec is a GPU-vectorised likelihood. 
    #  The fiducial log-likelihood (~ FFT0_gpu[1]) is multiplied by 1/dL, and the other one by 1/dGW.
    #  In the absence of gravity modification, dL = dGW. See also comments in file 'propagation.py'.
    #    
    #  ENTER MODIFICATIONS TO LUMINOCITY DISTANCE DUE TO PROPAGATION EFFECTS in 'L_gpu_vec' below.
    #  THE FUNCTIONAL FORMS FOR THE MODIFIED GW LUMIN. DISTANCE MUST BE DEFINED IN THE FILE 'propagation.py': 
    #
    # [1] Multiply the FFT-ed waveform in the likelihood below with the product of all terms which should enter in           #     the dGW. For example: dGW1(..)*dGW2(..)*FFT_i_gpu[1]. dGW's must be defined in 'propagation.py'.
    #
    # [2] For redshift-only modifications multiply by dGW(z) appropriately defined in 'propagation.py'.
    #
    # [3] For frequency-dependent modidicatios use the vectorised functions "get_modified_damping" or   
    #     "get_modified_speed" (file: 'propagation.py') which evaluate the frequency grid in vectorised form. 
    #     
    # E.g: get_modified_damping(FFT0_gpu[0],...) * FFT_i_gpu[1]. This will apply the frequency-dependent damping  
    #       evaluated on the frequency grid 'FFT0_gpu[0]', on the vector of the FFT-ed waveform.
    #           
    #
    # [4] Below, we use as example a redshift-dependent damping through the function dGW_Xi                                 #     (for more, see file 'propagation.py'.)  
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
        L_gpu_vec = get_Likelihood( (dL(z)**(-1))*FFT0_gpu[1], (dGW_Xi(Xi, z)**(-1))*FFT_i_gpu[1], FFT_i_gpu_0_j_noised) # vectorised likelihood
         
        
        temp_gpu = int_gpu(FFT_i_gpu[0][:], L_gpu_vec) # integrate over frequenciesc in the likelihood
        temp_cpu = temp_gpu.get()                      # get the likelihood vector in cpu form 
        
        
        log_likelihood_j = temp_cpu 
        
        return log_likelihood_j                        # return log(likelihood)
##################


###### This is a more recent noise function according to XXXX. It can be implemented in a similar fashion
# as the noise above within an ElementWise kernel within the 'get_noise' function above. 
#def noise_2(f):
#    L    = 2.5*10**11    #cm
#    fs   = 19.09*10**(-3) #Hz
#    A    = 9*10**(-45)                                                           
#    alpha= 0.171
#    beta = 292
#    kappa= 1020
#    gamma= 1680
#    fkappa= 0.00215
#    Poms = (1.5*10**(-9))**2    #cm^2/Hz
#    Pacc = ((3.0*10**(-13))**2)*(1 + (0.4*10**(-3)/f)**2)  #cm^2 Hz^3
#    Pn   = Poms/(L**2) + 2*(1 + (np.cos(f/fs)**2))*Pacc/(((2*Pi*f)**4)*(L**2))   #1/Hz
#    Sn   = (10/3)*Pn*(1 + (6/10)*(f/fs)**2)     #1/Hz
#    Sc   = A*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(kappa*f))*(1 + np.tanh(gamma*(fkappa-f)))
#    noise_2 = Sn + Sc
#    return noise_2