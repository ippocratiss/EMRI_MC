# Installation of EMRI_MC using Python venv

EMRI_MC was tested on python 3.11 at system with NVIDIA A100 using NVIDIA CUDA 12.2.0. and cupy 13.3.

## Installation using pip

(make sure NVIDIA CUDA is available, install it or load it by `module load CUDA/12.2.0` if software management is available at your system.)

(optional) create empty venv: `python3.11 -m venv ~/.venvs/emri_mc_prod` and activate it: `source ~/.venvs/emri_mc_prod/bin/activate`

clone the repository:

`git clone https://github.com/ippocratiss/EMRI_MC.git`

cd there and install it:

```
cd EMRI_MC
pip install .
```

## Manual Installation of EMRI_MC using Python venv

### Runtime environment preparation

Python - code was tested with #FIXME! python3.11. Make sure development package for given python version is installed too. Eg. at EL-like linux distros, `dnf install python3.11-devel`.

#### Ensure CUDA is present

eg. at cluster environments, load CUDA toolkit. [^1]

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

`pip install cupy emcee seaborn matplotlib pandas numpy emcee IPython ipykernel scipy h5py tqdm`

The install step `Running setup.py install for cupy ...` may take a considerable amount of time. 
The packages related to interactive Python, such as `IPython` and `ipykernel`, are only needed for the example Jupyter notebooks and are not required by the underlying library.

# Notes

[^1]: CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit
