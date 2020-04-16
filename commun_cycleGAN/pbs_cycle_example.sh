#!/bin/bash

#PBS -S /bin/bash
#PBS -N cycleGAN_example
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=12:ngpus=1:mem=32gb
#PBS -q gpuq
#PBS -P cycleGAN
#PBS -M paulin.brissonneau@student-cs.fr

# Go to the directory where the job has been submitted 
cd $PBS_O_WORKDIR

# Module load 
module load anaconda3/5.3.1

# Activate anaconda environment code
source activate tf-gpu

# Train the network
python main.py config_example_gpu.json