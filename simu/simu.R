#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

d=read.table(file=args[1],header=TRUE)

set.seed(d$seed)

x=rpois(d$n,d$lambda)

#y=runif(d$n)<d$p Pas besoin ici

res=data.frame(x=x)#,y=y)


write.table(res,file=args[2],row.names=FALSE)

#save(res,x,file=args[2])

#save.image(file=args[2])
