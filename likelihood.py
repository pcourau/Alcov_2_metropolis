import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_name",type=str)
parser.add_argument("parameters_name",type=str)
parser.add_argument("time_window",type=int)
parser.add_argument("nb_E_states",type=int)
parser.add_argument("nb_I_states",type=int)
parser.add_argument("-p","--people",type=int,default=2)
parser.add_argument("-e","--exact",type=bool,help="Type of inference used to calculate likelihood, default 1",default=1)
args=parser.parse_args()

import csv
import scipy.stats as stats

import pyAgrum as gum
from diffusionSIR import DiffusionSIR

with open(args.parameters_name) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    parameters = [k for k in csv_reader][0]
parameters = [float(k) for k in parameters]

#args.data_name denotes the name of the data file
#args.parameters denotes the name of the parameters file: eta0, eta, sigma, shape and scale of incubation time distribution (state Exposed E) and of contagious time distribution (state Infected I)

#if rate and shape parameters for the infection distribution are included, they must be transformed
E_distribution = stats.gamma.pdf([parameters[4]*k for k in range(1,args.nb_E_states+1)],a=parameters[3])
E_distribution = E_distribution[::-1]
I_distribution = stats.gamma.pdf([parameters[6]*k for k in range(1,args.nb_I_states+1)],a=parameters[5])
I_distribution = I_distribution[::-1]
mass_of_E_distrib = sum(E_distribution)
mass_of_I_distrib = sum(I_distribution)
if mass_of_E_distrib*mass_of_I_distrib != 0: #In case parameters of the gamma distribution become way too large
    E_distribution = E_distribution/sum(E_distribution)
    I_distribution = I_distribution/sum(I_distribution)
    parameters = parameters[0:5]
    parameters[3] = E_distribution
    parameters[4] = I_distribution

    bigOne=DiffusionSIR(people=["p"+str(k) for k in range(1,args.people+1)], T=args.time_window, nb_E_states=args.nb_E_states, nb_I_states = args.nb_I_states)
    bigOne.updateParameters(*parameters)


    stories=bigOne.readStories(args.data_name) 
    Loglik = bigOne.LL(stories,approx=args.exact)
else:
    Loglik = [-2000]
    
with open("LL_"+args.data_name,mode='w') as LL_file:
    LL_writer = csv.writer(LL_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    LL_writer.writerow(Loglik)
