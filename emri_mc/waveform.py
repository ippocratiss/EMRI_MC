#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
# This module defines useful functions needed for the MCMC run, including:
# the system of kludge equations, eqs()
# the waveform computation and its FFT on GPU/cuda, waveform(), FFT_gpu()
# noise function for the detector, noise()
# GPU-related functionality, e.g Bessel functions defined for cuda 
# possible modifications to the GW luminosity distance due to physics beyond GR
# possible modifications to the damping of the GW amplitude due to physics beyond GR
####################################################################################

##################
import emri_mc.global_parameters.physics as g   # import global parameters and initial conditions
import numpy as np
import cupy as cp # GPU functionality 
import scipy.special as sc
from scipy.integrate import solve_ivp # ODE solvers etc.
import matplotlib.pyplot as plt  # plotting 
##################

################## int_gpu()
def int_gpu(x,y):   
    """
    Simple discrete integration routine.

    Parameters
    ----------
    x : 1D array
    y : 1D array
        
    Returns
    -------
    integral value: real number
     """
    y_max    = y[1:]
    y_min    = y[:-1]
    N        = len(x)
    dx       = (x[-1] - x[0])/(N-1)
    integral = (0.5*dx)*cp.sum(y_max + y_min)

    return integral
##################


################## eqs()
# Note: The initial values for the variables (Phi, alpha, gamma, e) at LSO are defined in global_parameters.py.
def eqs(t, y, M0, μ0, spin0):
    """
    Definition of system of kludge orbital equations as in Barrack&Cutler2004.
    Extend the function arguments in case needed to include other variables (eccentricity, angles etc.)

    Parameters
    ----------
    t : 1D array
    y : 1D array
    M0: central mass M0 [solar mass]
    μ0: orbiting mass M0 [solar mass]
    spin0: spin of central black hole [G/c]

    Returns
    -------
    the l.h.s of the system of ODE equations
     """
    
    # Vector of orbital variables: Phase, frequency, angles γ and α
    Φ, ν, e, γ, α = y[0], y[1], y[2], y[3], y[4] 
    
    π    = g.Pi
    M    = M0*g.Ms              # Restore units in central mass
    μ    = μ0*g.Ms              # Restore units in orbiting mass      
    spin = spin0*(g.G/g.c)        # Restore units in spin
    
    x    = (2*π*g.G*M*ν)/g.c**3  # Post-Newtonian parameter
    
    ###### System of ODEs for {Φ, ν, e, γ, α} ######
    dPhi_dt = 2*π*ν
    #
    dnu_dt = (96/(10*π))*((g.c**6/g.G**2)*μ/M**3)*(x**(11/3))*(1 - e**2)**(-9/2)*((1 - e**2)*(1 + (73/24)*e**2 + (37/96)*e**4) + x**(2/3)*((1273/336) - (2561/224)*e**2 - (3885/128)*e**4 - (13147/5376)*e**6) - x*(g.c/g.G*spin)*np.cos(g.λ)*(1 - e**2)**(-1/2)*((73/12) + (1211/24)*e**2 + (3143/96)*e**4 + (65/64)*e**6)) 
    #
    dgamma_dt = 6*π*ν*x**(2/3)*((1 - e**2)**(-1))*(1 - 2*(x**(1/3))*(g.c/g.G*spin)*np.cos(g.λ)*(1 - e**2)**(-1/2)+ (1/4)*x**(2/3)*(1 - e**2)**(-1)*(26 - 15*e**2))
    #
    de_dt = -(e/15)*((g.c**3/g.G)*μ/M**2)*(x**(8/3))*((1 - e**2)**(-7/2))*((1 - e**2)*(304 + 121*e**2)+ (1/7)*(x**(2/3))*(8831 - 28995*e**2 - (56101/8)*e**4) - 3*x*(g.c/g.G*spin)*np.cos(g.λ)*((1-e**2)**(-1/2))*(1364 + (5032/3)*e**2 + (263/2)*e**4)) 
    #                       
    dalpha_dt = 4*π*ν*x*(g.c/g.G*spin)*(1 - e**2)**(-3/2)
    #############################
    
    return [dPhi_dt, dnu_dt, de_dt, dgamma_dt, dalpha_dt] # returns the r.h.s of ODEs
##################


################## compute_orbit()
def compute_orbit(p, x, t_min = g.t_min, t_max = g.t_max, solver = g.solver0):
    """
    Computes the solution of the orbital equations using Python's solve_ivp(). The set of
    equations is defined in the function eqs(). The ODE solver can be replaced with any other. 

    Parameters
    ----------
    p : 1D array, parameter values supplemented as args in the solve_ivp(). p = [M, μ, spin, ...]
    t_min, t_max: minimum and maximum integration time. Default values in 'global_parameters.py'.
    solver: Integration routine. 

    Returns
    -------
    2D array, [time grid, solution]
    time grid, 1D array of real values
    solution = [Φ, ν, e, γ, α], where each component of array is a 1D vector of real values 
    """   
    points = int(np.floor(0.1*np.abs(t_max-t_min)))  # number of grid points for integration. 
    t_span = np.linspace(t_max,  t_min, points)      # time grid for integration
    x0 = x                                           # vector with initial conditions 
    args = [p[0],p[1],p[2]]                          # vector with parameter values [M, μ, S, ... ]
    
    # call the ODE solver 
    sol0   = solve_ivp(eqs, (t_max, t_min), x0, method=solver, t_eval = t_span, args=args, rtol=g.rtol, atol=g.atol)
    if sol0.success: # check that the solution is succesful 
        return [sol0.t, sol0.y]
    else:
        return 0 # Return 0 if solution is not found. This is useful for the MCMC iterations in order to skip a step.

################## waveform()
def waveform(nmax, M0, μ0, spin0, t, y):  
    """
    Computation of the waveform h = h(t) given a solution of the orbital equations,
    including the LISA response function and redshift corrections.

    Parameters
    ----------
    nmax: integer, maximum number of overtones to compute
    M0: central mass M0 [solar mass]
    μ0: orbiting mass M0 [solar mass]
    spin0: spin of central black hole [XXX]
    t : 1D array, time grid
    y : 5D array, grid with solutions of orbital variables (y[0]...y[4])

    Returns the 2D array of the waveform as a grid in time (t) and amplitude (h)
    -------
     """

    M    = M0*g.Ms       # Restore units in central mass
    μ    = μ0*g.Ms       # Restore units in orbiting mass 
    spin = spin0*(g.G/g.c) # Restore units in spin

    π = g.Pi
    λ = g.λ
    
    # Orbital variables
    Φ, ν, e, γ, α = y[0], y[1], y[2], y[3], y[4]
    
    # Default values for angles in the binary system. R = 499 seconds = 1 UA. 
    θS, φS, θK, φK, φ0, T1year, R = π/4, 0, π/8, 0, 0, 3.127*10**7, 499 
    
    
    x = cp.asarray((2*π*g.G*M*ν)/g.c**3,dtype=np.float64)
    A = cp.asarray(x**(2/3)*(g.G*μ/g.c**2),dtype=np.float64)
    t = cp.asarray(t,dtype=np.float64)
    α = cp.asarray(α,dtype=np.float64)
    γ = cp.asarray(γ,dtype=np.float64)
    e = cp.asarray(e,dtype=np.float64)
    Φ = cp.asarray(Φ,dtype=np.float64)
    
    
    # Definition of quantities needed for the waveform including the LISA response function.
    θL = cp.arccos(cp.cos(θK)*cp.cos(λ) + cp.sin(θK)*cp.sin(λ)*cp.cos(α)) 
    φL = cp.arctan2((cp.sin(θK)*cp.sin(φK)*cp.cos(λ) - cp.cos(φK)*cp.sin(λ)*cp.sin(α) - cp.sin(φK)*cp.cos(θK)*cp.sin(λ)*cp.cos(α)), (cp.sin(θK)*cp.cos(φK)*cp.cos(λ) + cp.sin(φK)*cp.sin(λ)*cp.sin(α) - cp.cos(φK)*cp.cos(θK)*cp.sin(λ)*cp.cos(α)))
    
    Ln = cp.cos(θS)*cp.cos(θL) + cp.sin(θS)*cp.sin(θL)*cp.cos(φS - φL)
    Sn = cp.cos(θS)*cp.cos(θK) + cp.sin(θK)*cp.sin(θS)*cp.cos(φS - φK)

    β = cp.arctan2((Ln*cp.cos(λ) - Sn), (cp.sin(θS)*cp.sin(φS - φK)*cp.sin(λ)*cp.cos(α) + (Sn*cp.cos(θK) - cp.cos(θS))/(cp.sin(θK))*cp.sin(λ)*cp.sin(α)))
    γtilde = γ + β

    θ = cp.arccos(cp.cos(θS)/2 - (cp.sqrt(3)*cp.sin(θS)/2*cp.cos((2*π*t)/T1year - φS))) 
    φ = (2*π*t)/T1year + cp.arctan2((cp.sqrt(3)*cp.cos(θS) + cp.sin(θS)*cp.cos((2*π*t)/T1year - φS)), (2*(cp.sin(θS)*(cp.sin((2*π*t)/T1year - φS)))))
    ψ = cp.arctan2((cp.cos(θL)/2 - (cp.sqrt(3)*cp.sin(θL)*cp.cos((2*π*t)/T1year - φL))/2 - cp.cos(θ)*(cp.cos(θS)*cp.cos(θL) + cp.cos(φL - φS)*cp.sin(θS)*cp.sin(θL))), (0.5*(cp.sin(θL)*cp.sin(θS)*cp.sin(φL - φS)) - 0.5*(cp.sqrt(3)*cp.cos((2*π*t)/T1year)*(cp.cos(θL)*cp.sin(θS)*cp.sin(φS) - cp.cos(θS)*cp.sin(θL)*cp.sin(φL))) - 0.5*(cp.sqrt(3)*cp.sin((2*π*t)/T1year)*(cp.cos(θS)*cp.cos(φL)*cp.sin(θL) - cp.cos(φS)*cp.cos(θL)*cp.sin(θS)))))

    F1plus  = ((1 + cp.cos(θ)**2)*cp.cos(2*φ)*cp.cos(2*ψ))/2 - cp.cos(θ)*cp.sin(2*φ)*cp.sin(2*ψ)
    F1cross = ((1 + cp.cos(θ)**2)*cp.cos(2*φ)*cp.sin(2*ψ))/2 + cp.cos(θ)*cp.sin(2*φ)*cp.cos(2*ψ)

    F2plus  = ((1 + cp.cos(θ)**2)*cp.sin(2*φ)*cp.cos(2*ψ))/2 + cp.cos(θ)*cp.cos(2*φ)*cp.sin(2*ψ)
    F2cross = ((1 + cp.cos(θ)**2)*cp.sin(2*φ)*cp.sin(2*ψ))/2 - cp.cos(θ)*cp.cos(2*φ)*cp.cos(2*ψ)
    
    z = e
    A_plus_total = 0 # initial value for sum
    A_cross_total = 0 # initial value for sum


    # There is no cupy.jv() function for the Bessel functions. Therefore, we define them as follows:
    # (1) For n>-1 we can go with cuda libary function https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g643b020a4c66860acc8c0f0a76f7b67e
    # -> we'll wrapp simple CUDA-C/C++ code into ElementwiseKernel because there is no Python wrapper
    # (2) For n=-1 (the first iteration of loop below) - the function result is not defined for the cuda implementation so we have to use sci-py implementation - b0:

    # hack for first Bessel function on cude (first loop interation, en = -1)
    b0 = cp.asarray(sc.jv(-1, z.get()),dtype=np.float64)

    # Rest of Bessel functions for cuda
    get_all_bessels = cp.ElementwiseKernel(
        'int32 en, T x',
        'T jn_nm2, T jn_nm1, T jn_n, T jn_np1, T jn_np2',
        '''
           jn_nm2 = jn((en-2),x);
           jn_nm1 = jn((en-1),x);
           jn_n   = jn(en,x);
           jn_np1 = jn((en+1),x);
           jn_np2 = jn((en+2),x);
        ''',
        'get_all_bessels'
    )

    get_an_bn_cn = cp.ElementwiseKernel(
        'int32 en, T e, T A, T Fi, T jn_nm2_nz, T jn_nm1_nz, T jn_n_nz, T jn_np1_nz, T jn_np2_nz',
        'T a_n, T b_n, T c_n',
        '''
           a_n = -en * A * ( jn_nm2_nz - 2 * e * jn_nm1_nz + (2.0/en) * jn_n_nz + 2 * e * jn_np1_nz - jn_np2_nz ) * cos(en*Fi);
           b_n = -en * A * pow((1 - pow(e,2.0)),(0.5)) * ( jn_nm2_nz - 2 * jn_n_nz + jn_np2_nz ) * sin(en*Fi);
           c_n = 2 * A * jn_n_nz * cos(en*Fi);
        ''',
        'get_an_bn_cn'
    )

    get_Aplus_A_cross = cp.ElementwiseKernel(
        'T Ln, T a_n, T b_n, T c_n, T ytilde ',
        'T Aplus_n, T Across_n',
        '''
           Aplus_n = -(1.0 + pow(Ln,2.0)) * ( a_n * cos(2.0 * ytilde) - b_n * sin(2.0 * ytilde) ) + (1.0 - pow(Ln,2.0)) * c_n;
           Across_n = 2.0 * Ln * (b_n * cos(2.0 * ytilde) + a_n * sin(2.0 * ytilde));
        ''',
        'get_Aplus_A_cross'
    )
 
    for n in range(1,nmax+1): # sums over orbital overtones n. Integer nmax is the maximum overtone to sum.
        n_z = cp.asarray((n*z),dtype=np.float64)
        jn_nm2_nz,jn_nm1_nz,jn_n_nz,jn_np1_nz,jn_np2_nz = get_all_bessels(n,n_z) # get Bessel functions. See above.
        
        if (n==1):  #fix the special case where cuda function is not giving what we need. See above.
            jn_nm2_nz = b0 
        
        a_n, b_n, c_n = get_an_bn_cn(n,e,A,Φ,jn_nm2_nz,jn_nm1_nz,jn_n_nz,jn_np1_nz,jn_np2_nz) # Coefficients entering the waveform.
        A_plus_n, A_cross_n = get_Aplus_A_cross(Ln,a_n,b_n,c_n,γtilde) # A_plus, A_cross
        A_plus_total += A_plus_n # Sums over eigenfrequencies up to nmax for A_plus.
        A_cross_total += A_cross_n # Sums over eigenfrequencies up to nmax for A_cross.

    hI  = (0.5*cp.sqrt(3))*(F1plus*A_plus_total + F1cross*A_cross_total) # waveform with detector response function I
    hII = (0.5*cp.sqrt(3))*(F2plus*A_plus_total + F2cross*A_cross_total) # waveform with detector response function II
 
    # total waveform with detector response function without the factor 1/distance in front of it. The prefactor is added conventionaly in the likelihood; the reason being that the prefactor may have parameters to be varied in the MCMC. See function iterate_mcmc() below.
    h   = hI + hII
    
    return [t, h] # Returns 2D array: [time grid, total waveform amplitude] 
##################


################## FFT_gpu()
def FFT_gpu(xt,yt,norm="forward"):
    """
    Performs the FFT on GPU/cuda.

    Parameters
    ----------
    xt : 1D array, time-domain values
    yt : 1D array, amplitude values h = h(t)
    norm: Optional parameter for the overal normalisation of the FFT. Defaul is "forward". See numpy documentation.

    Returns
    -------
    2D array: FFT transform in the form [frequency, asbolute of FFT amplitude].
     """
    
    N     = len(yt)                                        # length of waveform grid
    Y     = cp.fft.rfft(yt, norm=norm)                     # perform the FFT 
    Y_abs = cp.absolute(Y[:(N//2)+(N%2)])                  # absolute magnitude of complex values.
    freq  = cp.fft.fftfreq(len(xt), np.abs(xt[1] - xt[0])) # construct frequency grid
    freq  = freq[:(N//2)+(N%2)]                            # pick only the positive frequencies
    
    return [freq, Y_abs * (xt[0] - xt[-1])]  # returns the FFT as 2D grid: [frequencies, absolute magnitude of FFT]
##################


################## noise()
def noise(f):
    """
    Noise function of detector. For CPU only.

    Parameters
    ----------
    f : real number, frequency

    Returns
    -------
    real number, noise of detector
     """
    S_inst  = (1.22*10**(-51))*f**(-4) + (1.22*10**(-37))*f**2 + 2.12*10**(-41)
    S_exgal = 4.2*10**(-47)*f**(-7/3)
    S_gal   = 2.1*10**(-45)*f**(-7/3)
    S       = S_inst + S_exgal + S_gal     # total noise contribution
    return S
##################



################## compute_fiducial()
def compute_fiducial(x,p):
    """
    Computes the waveform and its FFT of the fiducial model around which we will perform the MCMC. 
    It calls the compute_orbit() to solve the system of orbital ODEs.

    Parameters
    ----------
    x : 1D array, initial conditions for the ODE solver
    p : 1D array, fiducial value for the parameters to be varies in the MCMC run
        
    Returns
    -------
    2D array, [ [time grid, amplitude of fiducial waveform], [frequency grid, FFT of amplitude of fiducial waveform] ] 
    """        
    #x0      = x # vector with initial conditions 
    p0       = [p[0],p[1],p[2]]  # vector with parameter values [M, μ, S, ... ]
    #sol0    = solve_ivp(eqs, (t_max, t_min), x0, method=solver0, t_eval = t_span, args=p0, rtol =rtol, atol=atol)
    sol0     = compute_orbit(p, x)
    WF0      = waveform(g.n_max, g.M0, g.μ0, g.spin0, sol0[0], sol0[1]) # compute waveform for fiducial model
    FFT0_gpu = FFT_gpu(WF0[0], WF0[1])                          # compute FFT of the fiducial model
    
    return [WF0, FFT0_gpu]
##################


################## plot_waveform()
def plot_waveform(x,y,hours):
    """
    Plots the waveform #hours before the plunge.

    Parameters
    ----------
    x : 1D array, time grid
    y : 1D array, grid with values for the waveform 
    hours: real number, hours before the plunge 
    Returns
    -------
    plot of the waveform in the sence y = y(x). The plotted waveform is not rescaled by the lum. distance ~ 1/D.
    """        
    plt.plot(x,y)
    #plt.plot(x,y,color='orange')
    hours = 1.    # hours to plots before the plunge at the LSO
    plt.xlim(g.t_max - 60*60*hours, g.t_max)
    plt.xlabel('time (sec)')
    plt.ylabel('h(t)',rotation=0) # h(t) here is NOT rescaled by 1/D.
    
##################
