#!/bin/bash
 
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=80GB
#PBS -l jobfs=5GB
#PBS -q gpuvolta
#PBS -P kr97
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/kr97
#PBS -l wd
  
module load pytorch/1.10.0

python3 LatentSpace_Prediction_training.py > /scratch/kr97/$USER/comp4560/$PBS_JOBID.log