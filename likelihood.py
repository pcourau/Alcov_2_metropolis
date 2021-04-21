import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_name",type=str)
parser.add_argument("parameters_name",type=str)
parser.add_argument("time",type=int)
parser.add_argument("--exact",type=bool,help="Type of inference used to calculate likelihood, default 1",default=1)
args=parser.parse_args()

import csv

with open(args.parameters_name) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    parameters = [k for k in csv_reader][0]
parameters = [float(k) for k in parameters]

#args.data_name denotes the name of the data file
#args.parameters denotes the name of the parameters file


import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from diffusionSIR import TimeIt,DiffusionSIR
gum.config["notebook","potential_visible_digits"]=7

bigOne=DiffusionSIR(people="AB",T=args.time)
bigOne.updateParameters(*parameters)


stories=bigOne.readStories(args.data_name) 

with open("LL_"+args.data_name,mode='w') as LL_file:
    LL_writer = csv.writer(LL_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    LL_writer.writerow(bigOne.LL(stories,approx=args.exact))
