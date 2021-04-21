n=250
lambda1=5.0
lambda2=7.0
p=0.7



for (rep in 1:50) {
	write.table(data.frame(n=n,lambda=lambda1,p=p,seed=floor(runif(1)*1e6)),file=paste0("test1_",rep,".in"),row.names=FALSE,quote=FALSE)
}
for (rep in 1:100) {
        write.table(data.frame(n=n,lambda=lambda2,p=p,seed=floor(runif(1)*1e6)),file=paste0("test2_",rep,".in"),row.names=FALSE,quote=FALSE)
}

