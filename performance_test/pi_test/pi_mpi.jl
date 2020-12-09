using MPI
function mpi_pi(NBIN::Int= 1000000000)::Float64
    MPI.Init()
    comm = MPI.COMM_WORLD
    myid = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    cpu1 = MPI.Wtime()

    my_step = 1/NBIN
    my_sum = 0
    for i in 0:nprocs:NBIN-1
        x = (i+0.5)*my_step;
        my_sum += 4 / (1 + x^2)
    end
    partial = my_sum*my_step
    println("Node $myid has partial value $partial")
    MPI.Barrier(comm)
    my_pi = MPI.Allreduce(partial, (x,y)->x+y, comm)

    cpu2 = MPI.Wtime()

    if myid == 0
        @show my_pi
        println("Execution timme (s) = $(cpu2-cpu1)")
    end
    return my_pi
end

if length(ARGS)>0
    mpi_pi(parse(Int, ARGS[1]))
else
    mpi_pi()
end
