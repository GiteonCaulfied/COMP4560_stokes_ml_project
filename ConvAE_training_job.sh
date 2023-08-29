#!/bin/bash
 
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=20GB
#PBS -l jobfs=10GB
#PBS -q gpuvolta
#PBS -P kr97
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/kr97
#PBS -l wd
  
module load pytorch/1.10.0

python3 ConvAE_training.py > /scratch/kr97/$USER/comp4560/$PBS_JOBID.log