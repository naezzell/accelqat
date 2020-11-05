# accelqat
We accelerate the QuantumAnnealingTools.jl package using Julia's equivalent of OpenMP, MPI, and CUDA.

## Setup on HPC

### julia

1. put both `QuantumAnnealingTools.jl` and `Qtbase.jl` in local directory
2. add julia package, (can also add to .bashrc like we did for MPI)

    `$ module load julia`

3. change package server 

    `$ export JULIA_PKG_SERVER=pkg.julialang.org`

4. start julia

    `$ julia --project=QuantumAnnealingTools.jl`

5. manually add `QTBase.jl` in package mode (by pressing `]` )

    `(QuantumAnnealingTools) pkg> add ~/path/to/QTBase.jl`

6. run unit test (better run in compute nodes)

    `julia> Pkg.test()` or `(QuantumAnnealingTools) pkg> test`

**Note:** 

- if fail with message like below, please run  `$ rm -rf ~/.julia/registries/General` (clean all registries) and try again from step 3.

    `ERROR: failed to clone from https://github.com/JuliaNLSolvers/NLSolversBase.jl.git, error: GitError(Code:ERROR, Class:Net, failed to resolve address for github.com: Name or service not known)`

- **packages' loading and updating should run on the login node, since compute nodes do not have access to the internet.**

### MPI for julia

see: [https://juliaparallel.github.io/MPI.jl/stable/configuration/](https://juliaparallel.github.io/MPI.jl/stable/configuration/)

1. add MPI package on julia (on login node)

    `(@v1.4) pkg> add MPI`

2. build package with system MPI 

    `$ julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.build("MPI"; verbose=true)`

3. run mpi julia script (on compute nodes)

    `$ mpiexec -n 4 julia --project pi_mpi.jl`

    [pi_mpi.jl](Setup%20repo%20on%20HPC%20ab4795a0a26742008190ca88709a937f/pi_mpi.jl)