#!/bin/bash
#
#SBATCH --job-name=ac_baseline
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=600:00
#SBATCH --mem-per-cpu=4000

source /home/users/gangus/analytic-continuation/.env/bin/activate
ml python/3.6
ml py-pytorch/1.0.0_py36
ml py-numpy/1.14.3_py36

run /home/users/gangus/analytic-continuation/experiments/coefficient_regression/baseline
run /home/users/gangus/analytic-continuation/experiments/coefficient_regression/baseline_regularized
