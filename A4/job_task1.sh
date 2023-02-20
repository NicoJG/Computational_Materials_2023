#!/usr/bin/env bash
#SBATCH -A C3SE2023-2-2 # Project
#SBATCH -J A4Task1 # Name of the job
#SBATCH -N 1 # Use 1 node
#SBATCH -n 8 #
#SBATCH -t 00:10:00 # Maximum time
#SBATCH -o task1output/std.out # stdout goes to this file
#SBATCH -e task1output/err.out # stderr goes to this file

# Unload all modules, to make sure no incompatibilities arise
module purge

# Load desired modules
module load GPAW/22.8.0-foss-2022a

# Run program
srun gpaw python task1.py