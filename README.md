# Alcov_2_metropolis
This repository is dedicated to the analysis of the results of the [Alcov-2 project](https://www.cnrs.fr/fr/alcov2-une-enquete-de-grande-ampleur-pour-letude-de-la-transmission-de-sars-cov2-au-sein-des-foyers), following (in this branch) the diffusion SIR model from model from G.Nuel : https://codimd.math.cnrs.fr/uhr5RMapRDKyiRhf7eJnkg?view#Semi-Markov-individual-centered-SIR-model-work-in-progress.

The architecture is as follows: in the simu folder you will find a program to generate data according to the model.
The Keystone file then follows a Metropolis algorithm to fit parameters to simulated data. The original plan was to use the Makefile to parallelize the calculation of the likelihood. All the needed things to do this are still present but we switched to the empirically more effective strategy to group all simulations in a single file.

You will require the [pyAgrum](https://agrum.gitlab.io/pages/pyagrum-code-samples.html) library for Python.

