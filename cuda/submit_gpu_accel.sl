#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --gres=gpu:k40:1
#SBATCH --time=00:10:00
#SBATCH --output=gpu_accel_status.out
#SBATCH --account=anakano_429

julia --project=test try_gpu_accel.jl > gpu_accel_print.out
