## EMRI\_MC: A GPU-based code for Bayesian inference of EMRI waveforms

This is a code developed by **Ippocratis D. Saltas and Roberto Oliveri** to perfrom parameter inference for Extreme Mass Ratio Inspirals (EMRIs) including propagation effects. It is designed to run on GPUs. For more information we refer to:  

## 1. Modules

On top of Python's usual **numpy** and **scipy** libraries, to run the code one needs the following modules: 

1. **cuda/cupy**: https://nvidia.github.io/cuda-python/install.html



2. **emcee**: https://emcee.readthedocs.io/en/stable/user/install/



3. **multiprocessing**: https://pypi.org/project/multiprocessing/


4. **corner** or **seaborn** or anything similar: https://corner.readthedocs.io/en/latest/install/  https://seaborn.pydata.org/installing.html

The cuda/cupy library provides the GPU functionality, emcee is the library with the MCMC sampler, multiprocessing is needed for parallelisation, and the corner/seaborn for producing the corner plots. 

## 2. Main files

**global\_parameters.py**: This module defines the values of physical constants in cgs units, the parameters of the fiducial model, geometrical parameters and initial conditions of the binary system, parameters for the ODE solver (e.g., integration time window and grid resolution), and MCMC-related definitions. It also defines the maximum number of orbital overtones $n\_max$ in the computation of the waveform. A change in the number of the parameters in the MCMC requires adjusting the parameter vector in this module.  

**waveform.py**: This module defines the set of kludge ODE equations, the waveform generator, and some GPU-related functionality.

**mcmc.py**: This module defines the MCMC-related functions and the MCMC iterator.

**propagation.py**: This module defines the functions needed for the propagation of the GW wave through the cosmological background in the presence of any modified gravity effects. 

**main.ipynb**: Assuming all parameters and fiducial values are properly defined as explained earlier, 
this Jupyter notebook calls the main functions to initiate the MCMC run, 
using the package $\texttt{emcee}$. As a simple choice, we have currently set throughout the numerical computation 
the source location $\{\theta_S,\phi_S\} = \{\pi/4, 0\}$, 
the orientation of the spin $\{\theta_K,\phi_K\} = \{\pi/8, 0\}$, 
$\alpha_{LSO} = 0$, the angle $\lambda = \pi/6$, 
the initial eccentricity $e_{LSO}=0.3$, and $\gamma_{LSO} = 0$, $\Phi_{LSO} = 0$, 
for the respective initial conditions. 
These can be straighforwardly modified in the file waveform.py.

## 3. Running the code

Running the code is particularly simple. Placing all files in the same folder, and setting up all parameters as explained above, one starts the notebook main.ipynb, and executes the cells. The first cell computes the fiducial model, and the second starts the MCMC run around the chosen fiducial. The MCMC results are stored in a text file - please make sure the path is defined appropriately. 
