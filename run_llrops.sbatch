#!/bin/bash

#SBATCH --job-name=llrops
#SBATCH --nodes=3
#SBATCH --ntasks=96
#SBATCH --time=24:00:00
#SBATCH --output=logs/llrops_%j.out
#SBATCH --error=logs/llrops_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

uv run mpirun -n 96 python -m llrops run configs/llrops_reflector_bias_adjustment_detailed.yml --mpi

echo "End time: $(date)"
