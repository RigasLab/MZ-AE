# Mori Zwanzig Autoencoder MZ-AE

This repository contains code for training Mori Zwanzig Autoencoder framework as described in "Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder" (10.1098/rspa.2024.0259).

## Getting started

To install the required libraries, create a virtual environment using your preferable method and run the following command:

```
pip3 install -r requirements.txt
```

The source code is in directory "src/".

The training can be run through command provided in bashcript "jobs".

For training MZ-AE GFDc use the notebook (Notebooks/2DCyl/Case1_eval_notebook.ipynb)

To evaluate the trained models, notebooks are present in the repository "Notebooks".

The training dataset and the pre-trained models can be downloaded from 10.5281/zenodo.15029317.

## References

1. [Nektar++ ](https://www.sciencedirect.com/science/article/pii/S0010465515000533) is used for obtaining the 2DCylinder flow data.

