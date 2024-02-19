# Mori Zwanzig Autoencoder MZ-AE

This repository contains code for training Mori Zwanzig Autoencoder framework as described in "Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder", (here https://arxiv.org/abs/2310.10745).

## Getting started

To install the required libraries, create a virtual environment using your preferable method and run the following command:

```
pip3 install -r requirements.txt
```

The source code is in directory "src/".

The training can be run through command provided in bashcript "jobs". The corresponding data is available [here](https://1drv.ms/f/s!AvyaisSoiJmohT1KME46oTsjqwEp?e=5DfHTx).

To evaluate the trained models, notebooks are present in the repository "Notebooks".
The pre-trained models can be downloaded from [here].

This code can be easily adapted to use on cluster.

## References

1. Nektar++ [link](https://www.sciencedirect.com/science/article/pii/S0010465515000533) was used for obtaining the 2DCylinder flow data.

2. The Kuramoto Sivashisnky data was obtained using the solver from [link](https://arxiv.org/pdf/2106.06069.pdf).