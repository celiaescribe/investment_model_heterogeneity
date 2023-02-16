#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=gamma_no_demand
#SBATCH --output="outputs_script/gamma_no_demand.txt"
#SBATCH --error="outputs_script/error_gamma_no_demand.txt"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=100:00:00
#SBATCH --partition=cpu_shared
#SBATCH --account=gennn
#SBATCH --mem=MaxMemPerNode
## END SBATCH directives

## load modules
module load anaconda3/2020.11 #cuda/10.2

## execution
python run_cluster_heterogeneity.py -d 0.086 -n 0.15 --cvar 0.05 -p 0 --cap 10000 -i 140 -dir "gamma_heterogeneity_no_demand_uncertainty"