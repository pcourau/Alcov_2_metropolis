infiles= $(wildcard simu_$(TIME_WINDOW)*.out)
outfiles= $(subst simu_, LL_simu_, $(infiles))

all: $(outfiles)

LL_simu_%: simu_% param_metropolis.in likelihood.py
	/home/pc/anaconda3/bin/python3 likelihood.py $< param_metropolis.in	$(TIME_WINDOW)	$(NUMBER_E_STATES)	$(NUMBER_I_STATES)	--people=$(POPULATION_SIZE)

clean:
	rm -f *.out

