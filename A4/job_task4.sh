#!/usr/bin/env bash
#SBATCH -A C3SE2023-2-2 # Project
#SBATCH -J A4Task4 # Name of the job
#SBATCH -N 1 # Use 1 node
#SBATCH -n 8 #
#SBATCH -t 01:00:00 # Maximum time
#SBATCH -o task4output/std.out # stdout goes to this file
#SBATCH -e task4output/err.out # stderr goes to this file

# Unload all modules, to make sure no incompatibilities arise
module purge

# Load desired modules
module load GPAW/22.8.0-foss-2022a

# Run program
srun -u gpaw python task4.py