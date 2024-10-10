## EMRI\_MC: A GPU-based code for Bayesian inference of EMRI waveforms

EMRI\_MC performs parameter inference for Extreme Mass Ratio Inspirals (EMRIs) based on the kludge formalism, including propagation effects due to modified gravity. It is designed to make use of GPU parallelisation for higher computational efficiency. For more information we refer to:  https://arxiv.org/abs/2311.17174

- [Authors](#Authors)
- [Installation](#Installation)
- [Modules](#Modules)
- [Execution](#Execution)

## Authors
- Main authors: Ippocratis D. Saltas and Roberto Oliveri 

- Collaborators: Josef Dvoracek and Stephane Ilic

The code was developed predominantly by Ippocratis D. Saltas and Roberto Oliveri, with the invaluable contributions of Stephane Ilic and Josef Dvoracek on MCMC and parallelisation aspects. 

## Modules

**global\_parameters.py**: This module defines the values of physical constants in cgs units, the parameters of the fiducial model, geometrical parameters and initial conditions of the binary system, parameters for the ODE solver (e.g., integration time window and grid resolution), and MCMC-related definitions. It also defines the maximum number of orbital overtones $n\_max$ in the computation of the waveform. A change in the number of the parameters in the MCMC requires adjusting the parameter vector in this module.  

**waveform.py**: This module defines the set of kludge ODE equations, the waveform generator, and some GPU-related functionality.

**mcmc.py**: This module defines the MCMC-related functions and the MCMC iterator, including the prior and log-probability function.

**propagation.py**: This module defines the functions needed for the propagation of the GW wave through the cosmological background in the presence of any modified gravity effects. 

**run_code.py**: This module starts the MCMC run, building on the parameter definitions in check global_parameters.py.

**main.ipynb**: Assuming all parameters and fiducial values are properly defined as explained earlier, 
this Jupyter notebook calls the main functions to initiate the MCMC run, 
using the package $\texttt{emcee}$. As a simple choice, we have currently set throughout the numerical computation 
the source location $\{\theta_S,\phi_S\} = \{\pi/4, 0\}$, 
the orientation of the spin $\{\theta_K,\phi_K\} = \{\pi/8, 0\}$, 
$\alpha_{LSO} = 0$, the angle $\lambda = \pi/6$, 
the initial eccentricity $e_{LSO}=0.3$, and $\gamma_{LSO} = 0$, $\Phi_{LSO} = 0$, 
for the respective initial conditions. 
These can be straighforwardly modified in the code. 

## Installation 

On top of Python's usual **numpy** and **scipy** libraries, to run the code one needs the following modules: 

1. **cuda/cupy**: https://nvidia.github.io/cuda-python/install.html

2. **emcee**: https://emcee.readthedocs.io/en/stable/user/install/

3. **corner** or **seaborn** or anything similar: https://corner.readthedocs.io/en/latest/install/  https://seaborn.pydata.org/installing.html

The cuda/cupy library provides the GPU functionality, emcee is the library with the MCMC sampler, multiprocessing is needed for parallelisation, and the corner/seaborn for producing the corner plots. 

EMRI_MC was tested on Python 3.11, on a system with NVIDIA A100 and using NVIDIA's cuda 12.2.0. and cupy 13.3.

## Manual Installation of EMRI_MC using Python venv

### Runtime environment preparation

The code was tested with Python 3.11. Make sure the development package for given Python version is installed too. Eg. at EL-like Linux distributions, `dnf install python3.11-devel`.

#### Ensure CUDA is present

For example, on a cluster environment load the cuda toolkit. [^1]

```
module load CUDA/12.2.0
```

#### Create venv

...and install recent `pip`..

```
python3.11 -m venv ~/.venvs/EMRI_MC
pip install --upgrade pip
```

#### Activate venv and install code dependencies

`source ~/.venvs/EMRI_MC/bin/activate`

`pip install cupy emcee matplotlib pandas numpy emcee IPython ipykernel scipy h5py tqdm`

The install step `Running setup.py install for cupy ...` may take a considerable amount of time. 
The packages related to interactive Python, such as `IPython` and `ipykernel`, are only needed for the example Jupyter notebooks and are not required by the underlying library.


[^1]: CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit


## Execution

Setting up all parameters as explained in the paper, one starts the notebook main.ipynb, and executes the cells (the code can also be executed through the terminal). This notebook serves as an example - it first computes and plots the fiducial model, and then starts the MCMC run around the chosen fiducial. The MCMC results are stored in a text file, please make sure the path is defined appropriately. 
