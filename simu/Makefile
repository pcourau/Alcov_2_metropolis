infiles=$(wildcard *.in)
outfiles=$(infiles:.in=.out)

all: $(outfiles)

%.out: %.in simu.R
	Rscript --vanilla simu.R $*.in $*.out

clean:
	rm -f *.out
