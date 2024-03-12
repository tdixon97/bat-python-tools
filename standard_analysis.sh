#!/bin/bash

fit_name=$1

## fit reconstruction
python plot-reconstruction.py -c $1

./bin/filter_mcmc -f $1

## correlation matrix
python plot-correlations.py -c $1 -m 1

## projections
python plot-projections.py -c $1 -d "all"
python plot-projections.py -c $1 -d "types"
python plot-projections.py -c $1 -d "chan"
python plot-projections.py -c $1 -d "string"
python plot-projections.py -c $1 -d "floor"

### plot the activities

python plot-activities.py -c $1 -s 1 -o parameter -t M1
python plot-activities.py -c $1 -s 1 -o fit_range -t M1 
python plot-activities.py -c $1 -s 1 -o bi_range -t M1 

