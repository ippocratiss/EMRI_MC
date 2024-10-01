####################################################################################
# This module defines neccesary definitions of global parameters and initial conditions.
# The quantities in this module are kept fixed throughout the computation and are not varied in the MCMC run.

# Contents: 
# 1. Libraries
# 2. Physical constants in cgs units
# 3. Parameter values for fiducial model 
# 4. Geometrical parameters and initial conditions of the system.
# 5. Parameters needed for ODE solver
# 6. Parameters for the MCMC run !!! SET THE MCMC FILE PATH TO SAVE THE CHAIN
####################################################################################

###### Libraries
import numpy as np
######


###### Physical constants in cgs units.
Ms = 1.9892*10**(33)   # Solar mass, cgs
G  = 6.67428*10**(-8)  # Newton's G, cgs 
c  = 29979245800       # Speed of light, cgs
Pi = np.pi             # Constant π
H0 = 72/(3*10**19)     # Hubble parameters in sec
######


###### Parameter values for fiducial model. 
# Add/remove parameters as needed, but ensure all other relevant parts of the code are adjusted. 
μ0    = 10      # orbiting mass in units of solar mass [M_sun]
M0    = (10**6) # central BH mass in units of solar mass [M_sun]
spin0 = 0.1     # central BH spin [S/M^2]
Xi0   = 0.8     # modified gravity parameter modifying the GW lum. distance. 
#Xi   = Xi0     # comment this out in case the modified gravity parameter Xi is not fixed in the MCMC and use Xi0 above as the respective fiducial value.
z     = 0.01    # redshift of the source. Affects the luminosity distances. 

# p0: Vector with fiducial model parameters. The same set of parameters which will be varied under the MCMC. 
p0    = [M0, μ0, spin0, Xi0] 


###### Geometrical parameters and initial conditions of the system.

λ = Pi/6 # This is an angle which we fix throughout (fiducial and MCMC). 

# Initial values for phase Φ, angles γ and α, eccentricity e at the LSO. 
# Here they are set to be the same when solving the ODEs for both fiducial model and MCMC run. 
# If any of these parameters below are chosen to be varied under the MCMC, this line below 
# and other parts of the code need to be modified. 
Phi_LSO, gamma_LSO, alpha_LSO, e_LSO = 0, 0, 0, 0.3
# For more, see Barack&Cutler2004.
#######

###### Parameters needed for ODE solver

# nu_LSO: The initial condition of the orbital frequency (ν) at the LSO. It depends on the central mass and eccentricity.
nu_LSO = (c**3/(2*Pi*G*M0*Ms))*((1 - e_LSO**2)/(6 + 2*e_LSO))**(3/2) 

# x0: Vector with initial conditions for the ODE system. Here is defined to be equal to the LSO.
x0 = [Phi_LSO, nu_LSO, e_LSO, gamma_LSO, alpha_LSO ]  
# For more, see: Barack&Cutler2004. 

t_max   = 3.127*10**7                             # initial integration time in seconds (3.127*10**7 sec = 1 year).
t_min   = 0                                       # final integration time.
points  = int(np.floor(0.1*np.abs(t_max-t_min)))  # Grid points for integration. Defines resolution of waveform
t_span  = np.linspace(t_max,  t_min, points)      # time limits for integration
solver0 = 'RK23'                                  
# Choose between: 
#'BDF'(backward differentiation formula), 
#'LSODA' (ODEPACK), DOP853(Runge-Kutta 7th order), 
#'RK23'(Runge-Kutta 3rd order), 
#'RK45' (Runge-Kutta 5th order)
rtol    = 10**-10                                 # relative tolerance for ODE solver
atol    = 10**-10                                 # absolute tolerance for ODE solver
n_max   = 20                                      # maximum number of overtones to include in the waveform. 
######


###### Parameters for the MCMC run
filepath        = '.../MCMC.txt'                    # THIS IS THE PATH TO THE TEXT FILE WHERE TO SAVE THE MCMC RESULTS.
parameter_names = ['Mass M', 'Mass μ', 'spin', 'Ξ'] # Parameters varied under MCMC
Ndim    = 4                                         # Number of parameters to fit. Change this accordingly.
Nwalker = 2*Ndim                                    # Number of walkers for the MCMC run. By default, twice the Ndim.
Nsteps  = 2000                                      # Number of MCMC steps. Change this accordingly. 
######