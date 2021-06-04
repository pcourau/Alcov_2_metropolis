#Alcov_2_metropolis

This repository is dedicated to the analysis of the results of the Alcov-2 project, following (in this branch) the diffusion SIR model from model from G.Nuel : https://codimd.math.cnrs.fr/uhr5RMapRDKyiRhf7eJnkg?view#Semi-Markov-individual-centered-SIR-model-work-in-progress.

This model, called modele_1, adds to modele_0 an Exposed state which models the incubation period of the virus.

The architecture is as follows: the simu.R file allows the simulation of data, and the metropolis.py file implements all by itself the metropolis algorithm.
You will require the pyAgrum library for Python.
