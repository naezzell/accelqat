#!/bin/bash
#SBATCH --nodes=8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:00:59
#SBATCH --output=pi.out
#SBATCH --account=anakano_429

cat /proc/cpuinfo > cpuinfo.txt

BIN_NUMBER=1000000000
echo "***** Strong scaling *****"
mpiexec -n 1 julia --project pi_mpi.jl $BIN_NUMBER
mpiexec -n 2 julia --project pi_mpi.jl $BIN_NUMBER
mpiexec -n 4 julia --project pi_mpi.jl $BIN_NUMBER
mpiexec -n 8 julia --project pi_mpi.jl $BIN_NUMBER

echo "***** Weak scaling *****"
mpiexec -n 1 julia --project pi_mpi.jl $((BIN_NUMBER*1))
mpiexec -n 2 julia --project pi_mpi.jl $((BIN_NUMBER*2))
mpiexec -n 4 julia --project pi_mpi.jl $((BIN_NUMBER*4))
mpiexec -n 8 julia --project pi_mpi.jl $((BIN_NUMBER*8))
=======
usedcore=1
while [ $usedcore -lt 5 ]; do
  echo "***** origin *****"
  mpiexec -n 1 julia --project pi.jl
  echo "***** with mpi *****"
  mpiexec -n $usedcore julia --project pi_mpi.jl
  let usedcore*=2
done
