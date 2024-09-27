# Manual Installation of EMRI_MC using Python venv

## Runtime environment preparation

Python - code was tested with #FIXME! python3.11. Make sure development package for given python version is installed too. Eg. at EL-like linux distros, `dnf install python3.11-devel`.

### Ensure CUDA is present

eg. at cluster environments, load CUDA toolkit. [^1]

```
module load CUDA/12.2.0
```

### Create venv

...and install recent `pip`..

```
python3.11 -m venv ~/.venvs/EMRI_MC
pip install --upgrade pip
```

### Activate venv and install code dependencies

`source ~/.venvs/EMRI_MC/bin/activate`

#WIP, so far exact list of dependencies in unclear.

note: install step `Running setup.py install for cupy ...` can take long time.

`pip install cupy emcee seaborn matplotlib pandas numpy emcee IPython ipykernel scipy h5py`

# Notes

[^1]: CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit
