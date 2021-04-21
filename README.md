# Alcov_2_metropolis
This repository is dedicated to the analysis of the results of the [Alcov-2 project](https://www.cnrs.fr/fr/alcov2-une-enquete-de-grande-ampleur-pour-letude-de-la-transmission-de-sars-cov2-au-sein-des-foyers), following the diffusion SIR model from model from G.Nuel : https://codimd.math.cnrs.fr/uhr5RMapRDKyiRhf7eJnkg?view#Semi-Markov-individual-centered-SIR-model-work-in-progress.

The architecture is as follows: in the simu folder you will find a program to generate data according to the model.
The Keystone file then follows a Metropolis algorithm to fit parameters to the simulated or real data, using the Makefile to parallelize the calculation of the likelihood.

You will require the [pyAgrum](https://agrum.gitlab.io/pages/pyagrum-code-samples.html) library for Python.
