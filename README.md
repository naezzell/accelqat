# accelqat
In this repo, we explore high-performance computing (HPC) parallism in Julia. After learning the basics, we apply what we learned to a real-world application: gpu-accelerating ~~`QuantumAnnealingTools.jl`~~ [`OpenQuantumTools.jl`](https://github.com/USCqserver/OpenQuantumTools.jl), a package for simulating open quantum systems. We start with a few users' guides explaining how to set up Julia on Discovery, the HPC cluster at USC. In particular, we show how to set up and use MPI and CUDA, how to install OpenQuantumTools, and how to edit OpenQuantumTools in "develop mode."

For this project, we did the following
1. Wrote a simple code to calculate PI in Julia--our "hello world" of HPC just as we did in the CSCI 596 course (see [here](https://github.com/naezzell/accelqat/tree/main/performance_test))
2. Benchmarked the MPI implementation of PI, showing it's string and weak-scaling parallel efficiency (see [here](https://github.com/naezzell/accelqat/tree/main/performance_test/pi_test))
3. Profiled the OpenQuantumTools package when annealing a quantum spin-glass Hamiltonian, identifying the scrhodinger equation solver as the bottleneck (see [here](https://github.com/naezzell/accelqat/blob/main/performance_test/final_project_profiling.jl))
4. GPU-accelerated the solve_schrodinger bottleneck in OpenQuantumTools (see [here](https://github.com/naezzell/accelqat/tree/main/cuda) and [here](https://github.com/USCqserver/OpenQuantumTools.jl/blob/gpu-accel/src/QSolver/closed_system_solvers.jl))
5. Bechmarked the performance of GPU-accelerated schrodinger equation solver, showing a speed-up for n = 8 up to n = 10 qubits (see bottom of this page)

We found this a successful first crack at HPC with Julia, but as usual, there is always room for future work. Here, our future direction is straighforward: integrate our changes into the OpenQuantumTools.jl package properly and then attempt to GPU-accelerate the open quantum systems solvers. In other words, we only accelerated solve_schrodinger, but we also want to accerlate solve_lindblad, solve_redfield, etc...

## Setup on HPC

### Julia
add julia package, (can also add to .bashrc like we did for MPI)  
    `$ module load julia`

### install package
**Notice:** `QuantumAnnealingTools.jl`, `Qtbase.jl` now are open source and change name to [`OpenQuantumTools.jl`](https://github.com/USCqserver/OpenQuantumTools.jl), [`OpenQuantumBase.jl`](https://github.com/USCqserver/OpenQuantumBase.jl).
Please follow their instruction for installation, if fail, can try with different julia package server like step 2. 

1. put both `QuantumAnnealingTools.jl` and `Qtbase.jl` in local directory

2. change package server  
    `$ export JULIA_PKG_SERVER=pkg.julialang.org`

3. start julia  
    `$ julia --project=QuantumAnnealingTools.jl`

4. manually add `QTBase.jl` in package mode (by pressing `]` )  
    `(QuantumAnnealingTools) pkg> add ~/path/to/QTBase.jl`

5. run unit test (better run in compute nodes)  
    `julia> Pkg.test()` or `(QuantumAnnealingTools) pkg> test`

**Note:** 

- if fail with message like below, please run  `$ rm -rf ~/.julia/registries/General` (clean all registries) and try again from step 3.  
    `ERROR: failed to clone from https://github.com/JuliaNLSolvers/NLSolversBase.jl.git, error: GitError(Code:ERROR, Class:Net, failed to resolve address for github.com: Name or service not known)`

- **packages' loading and updating should run on the login node, since compute nodes do not have access to the internet.**

### MPI for Julia

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

## Experiments
### Pi code
#### Performance of using MPI on Pi

parallel efficiency

![image](https://user-images.githubusercontent.com/18574971/100689201-e0584b00-3338-11eb-9f51-3620d8d8908a.png)
![image](https://user-images.githubusercontent.com/18574971/100689212-e3ebd200-3338-11eb-80c6-c66bd13e4864.png)
![image](https://user-images.githubusercontent.com/18574971/100689222-e6e6c280-3338-11eb-84eb-996cb6bf6ffd.png)

see detailed data [HERE](https://github.com/naezzell/accelqat/tree/main/performance_test/pi_test/test_result/pi_pe_result)(including CPU info).

Notice: for the situation of one processor, mpi inter-processor communication did not happen, thus take less time then expect, and the data point is ignored in the plot.

### GPU accelerated package
Scalling on tf and n. As we can see the GPU acceleration get advantage when the input scale is large (n>=8). However it did not show obvious advantage when the time sequence (tf) is longer.

![image](https://user-images.githubusercontent.com/18574971/101561112-921afb80-3979-11eb-8b0a-b7cb7ed67812.png)
![image](https://user-images.githubusercontent.com/18574971/101561672-98f63e00-397a-11eb-8601-2701860f8b17.png)

The test code can be find here [accelqat/cuda/scaling_test.jl](https://github.com/naezzell/accelqat/blob/main/cuda/scaling_test.jl).  
The test result and relevent CPU information can be find here [accelqat/cuda/scaling_test_result/](https://github.com/naezzell/accelqat/tree/main/cuda/scaling_test_result)
