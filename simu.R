#This simulates stories.
args = commandArgs(trailingOnly = T) #parameter file name, population size, number of infectious states, time_max, number of simulations
population_size = as.integer(args[2])
nb_E_states = as.integer(args[3])
nb_I_states = as.integer(args[4])
time_window = as.integer(args[5])
nb_sims = as.integer(args[6])
#nb_stories_per_file = as.integer(args[7])

param=read.table(args[1],sep=",") #parameters: eta0, eta, sigma, dgam_shape, dgam_rate
eta0 =param[[1]]
eta  =param[[2]]
sigma=param[[3]] 
dgamE_shape = param[[4]]
dgamE_rate = param[[5]]
dgamI_shape = param[[6]]
dgamI_rate = param[[7]]
tauE = dgamma(1:nb_E_states,shape = dgamE_shape,rate = dgamE_rate)
tauE = tauE/sum(tauE)
tauI = dgamma(1:nb_I_states,shape = dgamI_shape,rate = dgamI_rate)
tauI = tauI/sum(tauI)


states=c("S",paste0("E",1:nb_E_states),paste0("I",1:nb_I_states),"R")
K=length(states)
tmp=1:K
names(tmp)=states
states=tmp

#Starting distribution
mu1=rep(0,K); mu1[1]=1

# rate=(1-eta0)*(1-eta)^delta_(t-1)
trans=function(N=population_size) {
  rate=(1-eta0)*(1-eta)^(1:N-1)
  one=rep(1,N)
  res=NULL
  res=rbind(res,c(states["S"],states["S"],rate))
  for (t in 1:nb_E_states)
    res=rbind(res,c(states["S"],states[paste0("E",t)],(1-rate)*tauE[t]))
  for (t in 1:nb_I_states)
    res=rbind(res,c(states["E1"],states[paste0("I",t)],one*tauI[t]))
  res=rbind(res,c(states["R"],states["R"],one))
  for (t in 2:nb_E_states)
    res=rbind(res,c(states[paste0("E",t)],states[paste0("E",t-1)],one))
  for (t in 2:nb_I_states)
    res=rbind(res,c(states[paste0("I",t)],states[paste0("I",t-1)],one))
  res=rbind(res,c(states["I1"],states["R"],one))
  colnames(res)=c("row","col",paste0("delta=",1:N-1))
  return(res)
}

pi=trans()
simul=function(n=time_window,N=population_size) {
  s=matrix(NA,n,N)
  for (j in 1:N) s[1,j]=1
  for (i in 2:n) {
    delta=sum(grepl("I",names(states)[s[i-1,]]))
    tmp=matrix(0,K,K)
    for (k in 1:nrow(pi)) tmp[pi[k,1],pi[k,2]]= ifelse(delta!=N,pi[k,3+delta],1) #If everyone is infected, we can put 1 in every nonzero location of the matrix
    for (j in 1:N) s[i,j]=sample(states,size=1,prob=tmp[s[i-1,j],])
  }
  obs=rep(NA,N)
  for (j in 1:N) {
    if (s[n,j]>states[paste0("E",nb_E_states)]) {
      obs[j]=min(which(s[,j]>states[paste0("E",nb_E_states)]))
      if (runif(1)<0*(1-sigma)) obs[j]=NA
    }
  }
  return(list(n=n,start=obs))
}


#nb_files = nb_sims%/%nb_stories_per_file
#for (repetition in 1:nb_files){
#  stories = NULL
#  for (bisrepetita in 1:nb_stories_per_file){
#    stories = rbind(stories,simul()[[2]])
#  }
#  write.table(stories,paste0("simu_",time_window,"_",repetition,".out"),row.names = F,quote=F,col.names = F,sep = ",")
#}
stories = NULL
for (repetition in 1:nb_sims){
  stories = rbind(stories,simul()[[2]])
}
write.table(stories,paste0("simu_",time_window,".out"),row.names = F,quote=F,col.names = F,sep = ",")
