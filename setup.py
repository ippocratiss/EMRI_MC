from setuptools import setup,find_packages

setup(
    name='emri_mc',
    version='0.0.2',
    description='A GPU-based code for Bayesian inference of EMRI waveforms',
    url='https://github.com/ippocratiss/EMRI_MC',
    author='Ippocratis Saltas, Roberto Oliveri',
    author_email='saltas@fzu.cz',
    license='Attribution-NonCommercial-ShareAlike 4.0 International',
    packages=find_packages(),
    install_requires=['cupy',
                      'numpy',
                      'scipy',
                      'tqdm',
                      'matplotlib',
                      'pandas',
                      'emcee',
                      'IPython',
                      'ipykernel',
                      'h5py'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Other/Proprietary License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.11',
    ],
)

