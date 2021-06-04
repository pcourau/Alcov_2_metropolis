import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-pn","--parameters_name",type=str, default="param0.in")
parser.add_argument("-tw","--time_window",    type=int, default=30)
parser.add_argument("-nE","--nb_E_states",    type=int, default=10)
parser.add_argument("-nI","--nb_I_states",    type=int, default=10)
parser.add_argument("-p", "--people",         type=int, default=2)

group = parser.add_mutually_exclusive_group()
group.add_argument("-ns","--nb_simulations",  type=int, default=100,help="Size of the simulated dataset") #Either enter a number of simulations or a data file
group.add_argument("-df","--data_file",       type=str, help="Data file to use to calculate the likelihood")

parser.add_argument("-ni","--nb_iterations",  type=int, default=2,help="Number of iterations of the metropolis algorithm")
parser.add_argument("-sm","--sample_size",    type=int, default=2,help="Number of points you simultaneously run the metropolis algorithm on")
parser.add_argument("-st","--step_size",      type=float,default=0.3,help="Size of the random gaussians added to the parameters")
parser.add_argument("-e", "--exact",          type=bool, help="Type of inference used to calculate likelihood, default 1",default=1)
args=parser.parse_args()

nb_param = 7 #These are: eta0, eta, sigma, shape of E distribution, rate of E distribution, shape of I distribution, rate of I distribution

if args.data_file == None:
    import subprocess
    import os
    subprocess.call(["/usr/bin/Rscript", os.getcwd()+"/simu.R", args.parameters_name, str(args.people), str(args.nb_E_states), str(args.nb_I_states), str(args.time_window), str(args.nb_simulations)])
    datafile = "simu_"+str(args.time_window)+".out"
else:
    datafile = args.data_file


import scipy.stats as stats
import numpy as np

import pyAgrum as gum
from diffusionSIR import DiffusionSIR

def Transform(list_parameters):#We transform these parameters so that we can use the Metropolis algorithm with a normal law
    if (len(list_parameters) < nb_param):
        raise(ValueError,"Incorrect nb of parameters in Transform")
    return np.append(np.log(list_parameters[0:2]/(1-list_parameters[0:2])), np.log(list_parameters[2:]))


def Untransform(theta):#to get the parameters back in a usable form for the model
    if (len(theta)<nb_param):
        raise(ValueError,"Incorrect nb of parameters in Untransform")
    return np.append(np.exp(theta[0:2])/(1+np.exp(theta[0:2])),np.exp(theta[2:]))

eta0        = np.random.random(args.sample_size)
eta         = np.random.random(args.sample_size)
sigma       = np.random.random(args.sample_size)
dgamE_shape = np.random.exponential(size = args.sample_size)
dgamE_rate  = np.random.exponential(size = args.sample_size)
dgamI_shape = np.random.exponential(size = args.sample_size)
dgamI_rate  = np.random.exponential(size = args.sample_size)
param = np.array([eta0, eta,sigma, dgamE_shape, dgamE_rate, dgamI_shape, dgamI_rate]) #careful ! This has shape (7,sample_size)


bigOne=DiffusionSIR(people=["p"+str(k) for k in range(1,args.people+1)], T=args.time_window, nb_E_states=args.nb_E_states, nb_I_states = args.nb_I_states)
stories=bigOne.readStories(datafile) 


NUMBER_OF_ACCEPTED_STEPS = np.array([0]*args.sample_size)
LOGLIK                   = [-np.inf]*args.sample_size

for step in range(args.nb_iterations):
    for samp in range(args.sample_size):
        local_param = param[:,samp]

        #1: create new parameters
        new_local_param = Untransform(Transform(local_param) + np.random.normal(scale = args.step_size, size = nb_param))
        
        #rate and shape parameters for the infection distribution must be transformed into their distribution
        E_distribution = stats.gamma.pdf([new_local_param[4]*k for k in range(1,args.nb_E_states+1)], a= new_local_param[3])
        I_distribution = stats.gamma.pdf([new_local_param[6]*k for k in range(1,args.nb_I_states+1)], a= new_local_param[5])
        mass_of_E_distrib = np.sum(E_distribution)
        mass_of_I_distrib = np.sum(I_distribution)
        if mass_of_E_distrib*mass_of_I_distrib != 0: #In case parameters of the gamma distribution become way too large: skip the rest, go to next step
            E_distribution = E_distribution/np.sum(E_distribution)
            I_distribution = I_distribution/np.sum(I_distribution)
            parameters = list(new_local_param[0:5])
            parameters[3] = E_distribution
            parameters[4] = I_distribution

            #2: Calculate loglikelihood
            bigOne.updateParameters(*parameters)
            NEW_LOGLIK = bigOne.LL(stories,approx=args.exact)[0]

            #3: Decide whether to keep the new parameters or not
            if NEW_LOGLIK > LOGLIK[samp]:
                LOGLIK[samp] = NEW_LOGLIK
                param[:,samp] = new_local_param
                NUMBER_OF_ACCEPTED_STEPS[samp] += 1
            elif np.random.random() < np.exp(NEW_LOGLIK - LOGLIK[samp]):
                LOGLIK[samp] = NEW_LOGLIK
                param[:,samp] = new_local_param
                NUMBER_OF_ACCEPTED_STEPS[samp] +=1
    print("Done step: "+str(step))

print("Loglikelihood: "+str(LOGLIK))
print("Parameter values: "+str(param))
print("Percent accepted steps: "+str(NUMBER_OF_ACCEPTED_STEPS/args.nb_iterations))
