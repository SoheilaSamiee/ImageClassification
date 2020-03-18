#!/bin/bash
#SBATCH --job-name=uBurstProp
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --mem=100Gb
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=2                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH -o /network/home/samieeso/python_projects/uBurstProp/slurm-%j.out  # Write the log on tmp1

# 1. Load the environment
module load conda
source activate CondaEnv

# 2. Launch the job: Run the code
python MicroBurstProp_MNIST_classification.py