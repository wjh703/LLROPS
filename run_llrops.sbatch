#!/bin/bash

#SBATCH --job-name=llrops
#SBATCH --nodes=3
#SBATCH --ntasks=96
#SBATCH --time=24:00:00
#SBATCH --output=logs/llrops_%j.out
#SBATCH --error=logs/llrops_%j.err

set -euo pipefail

# Small LLR normal systems are faster and more stable without per-rank BLAS
# thread pools; MPI provides the forward-model parallelism.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

uv run mpirun -n 96 python -m llrops run configs/llrops_reflector_bias_adjustment_detailed.yml --mpi
# uv run mpirun -n 96 python -m llrops run configs/llrops_oc_residuals.yml --mpi
echo "End time: $(date)"
