# Manual Installation of EMRI_MC using Python venv

## environment preparation

### ensure CUDA is present

eg. at cluster environments, load CUDA toolkit. [^1]

```
module load CUDA/12.2.0
```

### Create venv

`python3.11 -m venv ~/.venvs/EMRI_MC`

### Activate venv

`source ~/.venvs/EMRI_MC/bin/activate`

### Install cupy and required packages into it

work in progress, so far exact list of dependencies in unclear.

`pip install cupy emcee corner`




# Notes

[^1]: CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit
