import argparse
parser = argparse.ArgumentParser()

parser.add_argument("parameters_name",type=str)
parser.add_argument("time_max",type=int,help = "Duration of the simulation")
parser.add_argument("infection_duration",type=int,help = "Number of I states")
parser.add_argument("-ns","--number_of_simulations",type=int,default=1,help="number of simulations")

group = parser.add_mutually_exclusive_group()
group.add_argument("-bpf","--base_population_filename",type=str,help = "The file must be a base population: a list of size infection_duration + 2 containing the number of people initially in any given state")
group.add_argument("-ps","--population_size",type=int,help = "Here the base population will be individuals all in state S")

args=parser.parse_args()

import numpy as np
import csv

number_of_states = args.infection_duration + 2
if args.population_size == None:
    if args.base_population_filename == None:
        raise ValueError("Please specify an initial population filename (with -bpf) or a population size (-ps)")
    else:
        with open(args.base_population_filename) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=",")
            initial_pop= [int(k) for k in csv_reader]

            test = len(initial_pop) - 3
            if test !=0:
                if test < 0:
                    raise Warning("Your initial population only has {} states, {} expected. Filling up with 0s".format(len(initial_pop),3))
                    initial_pop += [0]*(-test)
                elif test > 0:
                    raise ValueError("Your initial population has too many states: {} received, {} expected".format(len(initial_pop),3))
            pop_size = np.sum(initial_population)
else:
    pop_size = args.population_size
    initial_pop = [pop_size]+[0]*(number_of_states-1)
    
with open(args.parameters_name) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    parameters = [k for k in csv_reader][0]
    parameters = [float(parameters[k]) for k in range(len(parameters))]


## How to simulate a population
#Import model
import pyAgrum as gum
from diffusionSIR import DiffusionSIR,TimeIt

bigOne=DiffusionSIR(people=["p"+str(k) for k in range(pop_size)],T=args.time_max)
bigOne.updateParameters(*parameters)

bn = bigOne._model

# add nodes to find the law of the date of symptoms
def transition_matrix(k):
    """Gives the transition matrix for variable date_O_i (see below): if date_O_{i-1} is nonzero, date_O_i = date_O_{i-1}, else, it equals i if O_i = 1, else it equals 0."""
    answer = [[[1] + [0]*k] + [[0]*k + [1]]]
    for i in range(1,k):
        answer+= [[[0]*i + [1] + [0]*(k-i)] + [[0]*(k+1)]]
    return answer

for people in range(pop_size):
    bn.add(f"date_O_p{people}_1",2)
    bn.addArc(f"O_p{people}_1",f"date_O_p{people}_1")
    bn.cpt(f"date_O_p{people}_1")[:] = [[1,0],[0,1]]
    for k in range(2,args.time_max):
        bn.add(f"date_O_p{people}_{k}",k+1)
        bn.addArc(f"O_p{people}_{k}","date_O_p"+str(people)+"_"+str(k))
        bn.addArc(f"date_O_p{people}_{k-1}","date_O_p"+str(people)+"_"+str(k))
        bn.cpt(f"date_O_p{people}_{k}")[:] = transition_matrix(k)

#Posterior distribution:
with TimeIt("method 1:"):
    ie = gum.LazyPropagation(bn)
    ie.addJointTarget({f"date_O_p{people}_{args.time_max-1}" for people in range(pop_size)})
    postdist = ie.jointPosterior({f"date_O_p{people}_{args.time_max-1}" for people in range(pop_size)})[:]

postdist = postdist/np.sum(postdist) #Not needed in theory, just to correct rounding mistakes
postdist = np.ndarray.flatten(postdist)

#Simulation:
def p_uplets(k,n):
    """this will list for a given k all the p-uplets of [1,k]**n"""
    if n==1:
        return np.array([[i] for i in range(k)],dtype="object")
    else:
        previous_iter = p_uplets(k,n-1)
        return np.array([np.append(i,j) for i in previous_iter for j in range(k)],dtype="object")

list_possibilities = p_uplets(args.time_max,pop_size)
simulation = list_possibilities[np.random.choice([k for k in range(len(list_possibilities))],size=args.number_of_simulations,p=postdist)]
#This, with a uniform random number, uses the posterior distribution of date_O to generate a corresponding event
def switch0NA(k):
    """To ensure compatibility between Python and R"""
    if int(k)==0:
        return "NA"
    else:
        return str(k)

for k in range(len(simulation)):
    simulation[k] = [switch0NA(i) for i in simulation[k]]


##

with open("data_simulation.in",mode='w') as output_file:
    for rep in range(args.number_of_simulations):
        output_writer = csv.writer(output_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(simulation[rep])

