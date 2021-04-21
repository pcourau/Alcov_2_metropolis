infiles= $(wildcard simu/*.out)
outfiles= $(subst simu/test, loglikelihood_, $(infiles))

all: $(outfiles)

loglikelihood_%: simu/test% param0.in likelihood.R
	echo $<
	Rscript --vanilla likelihood.R param0.in $<

clean:
	rm -f *.out

