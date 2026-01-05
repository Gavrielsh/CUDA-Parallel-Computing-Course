#!/bin/bash

#############################################################################
# SLURM Configuration for CUDA Homework 1
#############################################################################

#SBATCH --job-name=cuda_ex1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=rtx3070
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#############################################################################
# Execution Steps
#############################################################################

echo "Job started on host: $(hostname)"
echo "Date: $(date)"

# 1. GPU Info
echo "GPU Info from nvidia-smi:"
nvidia-smi

# 2. FIX: Try to load module, but if fails, add PATH manually
echo "Attempting to setup CUDA environment..."

# Try loading module (suppress error if fails)
module load cuda 2>/dev/null || echo "Module 'cuda' not found via module load."

# MANUAL FIX: Add standard CUDA paths to environment
# This is usually where cuda sits if modules don't work
export PATH=/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# 3. Check if nvcc is found now
echo "Checking for nvcc..."
which nvcc
nvcc --version

if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: nvcc still not found!"
    echo "Available modules are:"
    module avail
    exit 1
fi

# 4. Compile the code
echo "Compiling ex1.cpp..."
nvcc -x cu ex1.cpp -o ex1_executable -O3

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 5. Run the executable
echo "----------------------------------------------------------------"
echo "Starting execution..."
./ex1_executable
echo "----------------------------------------------------------------"
echo "Job finished."