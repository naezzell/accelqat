#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:00:59
#SBATCH --output=pi_thread.out
#SBATCH --account=anakano_429

module load gcc/8.3.0
module load julia/1.5.2
export JULIA_NUM_THREADS=8
cat /proc/cpuinfo > cpuinfo.txt

usedcore=1
while [ $usedcore -lt 5 ]; do
  echo "***** origin *****"
  mpiexec -n 1 julia --project pi.jl
  echo "***** with multithreading *****"
  mpiexec -n $usedcore julia --project pi_threads.jl
  let usedcore*=2
done
