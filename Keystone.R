##### This file will perform an MCMC analysis using the script Makefile multiple times over to parallelize #####
nb_iteration = 1000
population_size = 2
time_window = 30
nb_infectious_states = 10
nb_sims = 100
nb_stories_per_file = 100
nb_files = nb_sims%/%nb_stories_per_file

step_size = 0.5

#First: generate the data
setwd("~/Documents/biblio/M2/stage/Code/")
#system(paste0("python3 simu.py param0.in ",time_window," ", nb_infectious_states," -ps ",population_size))
system(paste("Rscript simu.R param0.in",population_size,nb_infectious_states,time_window,nb_sims,nb_stories_per_file,sep=" "))

#Transform parameters so that normal laws are best suited
Transform = function(list_parameters){#We transform these parameters so that we can use the Metropolis algorithm with a normal law
  if (length(list_parameters)<nb_param) stop("Incorrect nb of parameters in Transform")
#  return(c(log(min(max(list_parameters[[1]]/(1-list_parameters[[1]]),10^(-200))),10^200), log(min(max(list_parameters[[1]]/(1-list_parameters[[1]]),10^(-32))),10^32) , log(list_parameters[[3]]),log(list_parameters[[4]]),log(list_parameters[[5]])))
  return(c(log(list_parameters[[1]]/(1-list_parameters[[1]])), log(list_parameters[[2]]/(1-list_parameters[[2]])) , log(list_parameters[[3]]),log(list_parameters[[4]]),log(list_parameters[[5]])))
}

Untransform = function(theta){#to get the parameters back in a usable form for the model
  if (length(theta)<nb_param) stop("Incorrect nb of parameters in Untransform")
  return(c(exp(theta[1])/(1+exp(theta[1])),exp(theta[2])/(1+exp(theta[2])),exp(theta[3]),exp(theta[4]),exp(theta[5])))
}


#Fix parameters
dgam_shape=1
dgam_rate =1
param=data.frame(eta0=0.01,eta=0.2,sigma=0.7,dgam_shape,dgam_rate)
nb_param = length(param)

startclock = Sys.time()
write.table(param,"param_metropolis.in",row.names = F,quote=F,col.names = F,sep = ",")

#system(paste0("make -s TIME_WINDOW=",time_window," NUMBER_INFECTIOUS_STATES=",nb_infectious_states," POPULATION_SIZE=",population_size))
#Using make allows us to parallelize the calculation beween different stories
system(paste0("/home/pc/anaconda3/bin/python3 likelihood.py simu_",time_window,".out param_metropolis.in ",time_window," ",nb_infectious_states," --people=",population_size))
system("sync")

#LOGLIK = 0
#for (n in 1:nb_files) LOGLIK = LOGLIK + read.table(paste0("LL_simu_",time_window,"_",n,".out"),sep = ",")[[1]]
LOGLIK = read.table(paste0("LL_simu_",time_window,".out"),sep = ",")[[1]]
time_one_iteration = Sys.time() - startclock
print(paste0("Estimated time for the full calculation: ",nb_iteration*time_one_iteration))

RANDOM_NUMBERS = rnorm(nb_iteration*nb_param,sd = step_size)

# Iterate
NUMBER_OF_ACCEPTED_STEPS = 0
for (rep in 1:nb_iteration) {
  new_param = Untransform(Transform(param) + RANDOM_NUMBERS[(nb_param*(rep-1)+1):(nb_param*rep)])
  write.table(as.list(new_param),"param_metropolis.in",row.names = F,quote=F,col.names = F,sep=",")
#  system(paste0("make -s TIME_WINDOW=",time_window," NUMBER_INFECTIOUS_STATES=",nb_infectious_states," POPULATION_SIZE=",population_size))
  system(paste0("/home/pc/anaconda3/bin/python3 likelihood.py simu_",time_window,".out param_metropolis.in ",time_window," ",nb_infectious_states," --people=",population_size))
  system("sync")
#  NEW_LOGLIK = 0
#  for (n in 1:nb_files) NEW_LOGLIK = NEW_LOGLIK + read.table(paste0("LL_simu_",time_window,"_",n,".out"),sep = ",")[[1]]
  NEW_LOGLIK = read.table(paste0("LL_simu_",time_window,".out"),sep = ",")[[1]]
  
  if (NEW_LOGLIK > LOGLIK) {
    LOGLIK = NEW_LOGLIK
    param = new_param
    NUMBER_OF_ACCEPTED_STEPS = NUMBER_OF_ACCEPTED_STEPS +1
  } else{
    if (runif(1) < exp(NEW_LOGLIK - LOGLIK)) {
      LOGLIK = NEW_LOGLIK
      param = new_param
      NUMBER_OF_ACCEPTED_STEPS = NUMBER_OF_ACCEPTED_STEPS +1
    }
  }
  print(paste0("Done step ",rep))
  print(paste0("Current likelihood: ",LOGLIK))
  print(paste0("Likelihood tested: ",NEW_LOGLIK))
  print(paste0("Percent of accepted steps: ",NUMBER_OF_ACCEPTED_STEPS/rep))
}

print(param)
print(NUMBER_OF_ACCEPTED_STEPS/nb_iteration)
print(paste0("Total time: ",(Sys.time() - startclock)/60))


system(paste0("rm -rf *simu_",time_window,"*"))

