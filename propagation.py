#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_parameters import * # import parameter values 
from waveform import *          # import kludge equations and ODE solvers
import numpy as np
import cupy as cp 
import math
import scipy.special as sc
from scipy import interpolate, optimize
from scipy.integrate import quad, quadrature, fixed_quad,odeint, solve_ivp # ODE solvers etc.


## dL()
def dL(z, Ωm0 = 0.29, ΩΛ0 = 0.71, H0 = 72/(3*10**19)):
    """
     This is the EM luminosity distance.

Parameters
----------
z0 : scalar, redshift to source
Ωm0, ΩΛ0, H0: Cosmological parameters today (z = 0)

Returns
-------
dL: scalar, luminosity distance to the source
     """
    c = 29979245800  # speed of light, [cgs]
    
    def dL_integrand(z0):        
        return (Ωm0*(1+z0)**3 + ΩΛ0)**(-0.5) 
    
    int_Ez = fixed_quad( dL_integrand, 0, z )[0]  # This is the z-integral of the cosmological quantity E(z). 
    
    return (c/H0)*(1+z)*int_Ez                    # We assume low redshifts in this derivation. 



##dGW()
def dGW_Xi(Xi,z):    
    """
     This is the GW luminosity distance under the redshift-dependent parametrisation
     of: Belgacem et al 2018 - PRD98 (2018) 023510
         Matos et al 2023 - arXiv: 2210.12174
        
Parameters
----------
Xi : scalar, free dimensionless parameter parametrising the modification of the GW lumin. distance

Returns
-------
dGW: scalar, GW luminosity distance to the source modified through the parameter Xi
    """    
    return dL(z)*(Xi + (1 - Xi)/(1 + z))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Modified damping contribution: (1+z)^(-α_M/2). See draft for details.
# Modified speed contribution: Exp[-i*k*ΔT]. See draft for details.
#
# If the modified damping term (α_M) and/or speed (α_T) are ONLY redshift-dependent, it is fine to define the 
# modified GW luminosity distance through a standard Python function as in the function 'dGW_Xi' here. 
#
# If the modified damping term (α_M) and/or speed (α_T) in the propagation equation is FREQUENCY-DEPENDENT, 
# use the functions below which efficiently evaluate the frequency grid in vectorised form. The functions 
# 'get_modified_damping' and 'get_modified_speed' must multiply the FFT-ed waveform in the likelihood function   
# 'L_gpu_vec' defined in the file 'mcmc.py'. See the file mcmc.py for more comments. 
# When calling these functions remember to call them with their arguments get_modified_damping(f, ...).
#
# Choose the functional form for the frequency-dependent damping and modified speed as desired below. As an  
# example, the following parametrisation is chosen below: ν = a0 + a1*f. 
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# DEFINE alpha_M with frequency-dependent functional form as needed. 
# 
get_modified_damping = cp.ElementwiseKernel(
        'T f, float64 z, float64 a0, float64 a1',
        'T damping_vec',
    
        ''' 
            float alpha_M = a0 + a1*f;
            
            damping_vec = pow((1.0 + z),(-alpha_M/2.0));
        ''',
    
        'get_damping' 
                    ) 

# DEFINE alpha_T with frequency-dependent functional form as needed. 'k' is the Fourier k associated to the 
# propagating wave. See draft.  
# dL is the EM luminosity distance which must be provided as a number in the arguments of 'get_modified_speed'.
# The real part in 'speed_vec' must be considered.
get_modified_speed = cp.ElementwiseKernel(
        'T f, float64 z, float64 k, float64 a0, float64 a1, float64 dL',
        'T damped_vec',
        ''' 
            float alpha_T =  a0 + a1*f;
            
            float ΔT = -alpha_T*dL/(1+z);
            
            speed_vec = exp[-I*k*ΔT]; 
        ''',
    
        'get_damping' 

                  )
