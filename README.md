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
    
### CUDA for Julia
- [CUDA repo](https://github.com/JuliaGPU/CUDA.jl)

0. add CUDA to .bashrc (or .modules)   
    `module load cuda`

1. add CUDA pkg on Julia (login node)   
    `pkg> add CUDA`
    
2. test with salloc (see [using GPUs on Discovery](https://carc.usc.edu/user-information/user-guides/high-performance-computing/discovery/using-gpus) )     
    `salloc --ntasks=2 --time=30:00 --gres=gpu:k40:1 --account=anakano_429`   
    `julia> using CUDA`    
    `julia> u0 = cu(rand(1000))`
    
3. use with DifferentialEquations (bonus if you want to try)   
    - [guide to follow](https://github.com/SciML/DiffEqGPU.jl#within-method-gpu-parallelism-with-direct-cuarray-usage)
    - Note: Change `using OrdinaryDiffEq, CuArrays, LinearAlgebra` --> `using DifferentialEquations, CUDA, LinearAlgebra` to use packages consistent with QuantumAnnealingTools
    - Note2: You can use `@time` tag with and without CUDA array `cu()` to show that CUDA speeds up the solution of the ODE
    
    ### Using Local Version of QuantumAnnealingTools (dev mode)
- [Julia Pkg docs](https://docs.julialang.org/en/v1/stdlib/Pkg/)
- [helpful question on StackOverflow](https://stackoverflow.com/questions/58098296/julia-be-sure-to-use-dev-version-of-a-package)

In this portion, we assume you have a local version of QuantumAnnealingTools located in some direcory that we'll call "localQAT." We'll walk through a specific example of adding a function called "solve_schrodinger_gpu" in localQAT/src/QSolver/closed_system_solvers.jl 

1. add solve_schrodinger_gpu to closed_system_solver.jl     
    ```
    function solve_schrodinger_gpu(A::Annealing, tf::Real; tspan = (0, tf), kwargs...)
        u0 = cu(build_u0(A.u0, :v))
        p = ODEParams(A.H, float(tf), A.annealing_parameter)
        update_func = function (C, u, p, t)
            update_cache!(C, p.L, p, p(t))
        end
        cache = get_cache(A.H)
        diff_op = DiffEqArrayOperator(cache, update_func = update_func)
        jac_cache = similar(cache)
        jac_op = DiffEqArrayOperator(jac_cache, update_func = update_func)
        ff = ODEFunction(diff_op, jac_prototype = jac_op)

        prob = ODEProblem{true}(ff, u0, float.(tspan), p)
        solve(prob; alg_hints = [:nonstiff], kwargs...)
    end
    ```
    
2. add CUDA dependency and expose solve_schrodinger_gpu to user   
Open the file localQAT/src/QuantumAnnealingTools.jl
- add the following lines 
    ```
    import CUDA:
    cu
    ```   
    and 
    ```
    export solve_schrodinger_gpu
    ```
where export can be added to list at the end with "solve_unitary," "solve_schrodinger," etc...

3. create a new Julia environment (I don't think it's possible to "override" local QuantumAnnealingTools using `free` command, so this it's easier to just make a new devlopment environment)
    `julia --project=QATest'   
    - Then add necessary libraries on login node such as 
    `pkg> add DifferentialEquations`   
    `pkg> add ~/path/to/QTBase.jl`   
    `pkg> add CUDA`   
    - Finally, add localQAT in "dev" mode
    `(QATest) pkg> dev ~/path/to/localQAT`
    
 4. Verify that solve_schrodinger_gpu is accessible
    `julia> solve_schrodinger_gpu`   
    which whould output the following if the function is loaded    
    `solve_schrodinger_gpu (generic function with 1 method)`     
    
    Note: If it doesn't work, try restarting Julia and loading again. You may have tried to load QuantumAnnealingTools before making the changes locally. 
    
5. If you want to run this command, you'll have to load with GPU, so check out CUDA for Julia above
    
