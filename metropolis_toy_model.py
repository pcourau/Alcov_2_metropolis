import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson

#Parameters:
SIGMA = 0.1
NUMBER_ITERATIONS = 50000
N=10

#DATA
with open("data_toy.csv") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    line_count = 0
    DATA = [k for k in csv_reader][0]
DATA = np.array([int(k) for k in DATA])

#Prior
#theta estimates the log of the parameter of the poisson distribution
def Prior_Distrib(theta):
    return 1/2 * (norm.pdf(theta*2)+norm.pdf(2*(theta-2*np.log(10))))

#Distribution of the steps: normal distribution with variance sigma**2
RANDOM_NORMAL = np.random.normal(size = NUMBER_ITERATIONS*N)*SIGMA

#Calculate score:
def LogLikelihood(theta,x=DATA):
    return np.array( [np.sum( np.log(poisson.pmf(x, np.exp(theta[k]))) ) for k in range(len(theta))] )

def Score(theta,x=DATA):
    return LogLikelihood(theta,x)+np.log(Prior_Distrib(theta))

#Initialize according to prior
discrete_scale = np.linspace(-3,7,100)
discretized_density = Prior_Distrib(discrete_scale)
discretized_density = discretized_density/np.sum(discretized_density)
theta0 = np.random.choice(a=discrete_scale,size=N,p=discretized_density)

CURRENT_SCORE = Score(theta0)

#Iterate:
ACCEPTED_STEPS = np.repeat(0,N)
for iters in range(NUMBER_ITERATIONS):
    if iters%100==0: print(iters)
    new_theta0 = theta0 + RANDOM_NORMAL[range(N*iters,N*(iters+1))]
    NEW_SCORE = Score(new_theta0)
    for k in range(N):
        if (NEW_SCORE[k] > CURRENT_SCORE[k]):
            theta0 = new_theta0
            CURRENT_SCORE[k] = NEW_SCORE[k]
            ACCEPTED_STEPS[k] += 1
        elif (np.log(np.random.uniform()) <= NEW_SCORE[k] - CURRENT_SCORE[k]):
            theta0 = new_theta0
            CURRENT_SCORE[k] = NEW_SCORE[k]
            ACCEPTED_STEPS[k] += 1


#plots
def Density(LIST,MIN,MAX,STEP):
    ''' Returns x and y lists to plot the density of LIST from a to b with a step of c'''
    MIN=MIN-2*STEP
    MAX = MAX+2*STEP
    list_x = np.linspace(MIN,MAX,int(1/STEP))
    list_p = np.repeat(0,int(1/STEP))
    for x_i in LIST:
        list_p[int((x_i-MIN)/(STEP*(MAX-MIN)))] += 1
    return (list_x,list_p/N)

PERCENT_ACCEPTED = ACCEPTED_STEPS/NUMBER_ITERATIONS
LIST_X,LIST_Y = Density(PERCENT_ACCEPTED,0,1.1,5/N)
plt.figure("fig1")
plt.plot(LIST_X,LIST_Y,color="r")
plt.xlabel("Percent of accepted steps")
plt.title("Distribution of the percent of accepted steps among replicates")
plt.show()

LIST_X,LIST_Y = Density(theta0,np.min(theta0),np.max(theta0),5/N)
LIST_X2 = np.linspace(-5,np.log(20)+5,N)
plt.figure("fig2")
plt.plot(LIST_X,LIST_Y,color="r")
plt.plot(LIST_X2,Prior_Distrib(LIST_X2),color="b")
plt.xlabel("theta")
plt.title("Posterior (red) and prior (blue) distributions of theta")
plt.show()
