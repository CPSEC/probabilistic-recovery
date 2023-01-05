#!/bin/bash

strategies=(0 1 2 3)
noises=(0 0.001 0.002 0.003 0.004 0.005)
time=`date +%s`
for noise in ${noises[@]}; do
	for strat in ${strategies[@]}; do
		./multiple_run.bash $strat ${noise} #> results/${main_f_name}
	done
done

