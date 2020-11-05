#!/bin/bash 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:00:59
#SBATCH --output=pi.out
#SBATCH --account=anakano_429

cat /proc/cpuinfo > cpuinfo.txt

usedcore=1
while [ $usedcore -lt 5 ]; do
  echo "***** origin *****"
  mpiexec -n 1 julia --project pi.jl
  echo "***** with mpi *****"
  mpiexec -n $usedcore julia --project pi_mpi.jl
  let usedcore*=2
done