##### This file will perform an MCMC analysis using the script Makefile multiple times over to parallelize #####
nb_iteration = 2
set.seed(53)

#First: generate the data
setwd("~/Documents/biblio/M2/stage/Code/simu/")
#system("Rscript --vanilla prepare.R")
#system("make") #If you need to generate all the Poisson variables

#Fix parameters
setwd("..")

param=data.frame(lambda1=1.0,lambda2=1.0)
write.table(param,"param0.in",row.names = F,quote=F)
system("make")
LOGLIK = 0
for (i in 1:2) for (j in 1:ifelse(i==1,50,100)){
  LOGLIK = LOGLIK + read.table(paste0("loglikelihood_test",i,"_",j,".out"))
}

RANDOM_NUMBERS = rnorm(nb_iteration*2)
# Iterate

for (rep in 1:nb_iteration) {
  new_param = abs(param + 2*RANDOM_NUMBERS[2*rep:2*rep+1]/(1+rep))
  write.table(new_param,"param0.in",row.names = F,quote=F)
  system("make")
  NEW_LOGLIK = 0
  for (i in 1:2) for (j in 1:ifelse(i==1,50,100)){
    NEW_LOGLIK = NEW_LOGLIK + read.table(paste0("loglikelihood_test",i,"_",j,".out"))
  }
  
  if (NEW_LOGLIK > LOGLIK) {
    LOGLIK = NEW_LOGLIK
    param = new_param
  } else{
    if (runif(1) < NEW_LOGLIK / LOGLIK) {
      LOGLIK = NEW_LOGLIK
      param = new_param
    }
  }
}


print(param)

