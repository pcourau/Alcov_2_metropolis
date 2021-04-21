#### This calculates the likelihood given parameters lambda and n  ####
args = commandArgs(trailingOnly=TRUE)
lambda = read.table(args[[1]],header=T)[[1]]
x=read.table(args[[2]],header=T)
loglik = sum(log(dpois(x = x[[1]],lambda = lambda)))
write.table(loglik,paste0("loglikelihood_",strsplit(args[[2]],"/")[[1]][2]),row.names = F,quote=F,col.names = F)
