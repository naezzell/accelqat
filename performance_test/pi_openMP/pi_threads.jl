# pi computation for multithreading using Threads macro

using Base.Threads
import Base.Threads.Atomic

function thread_pi(NBIN::Int=100000)::Float64
    step_size = 1.0/NBIN
    pi_sum = Atomic{Float64}(0.0)
    Threads.@threads for i in 1:NBIN
        x = (i+0.5)*step_size
        Threads.atomic_add!(pi_sum, 4.0/(1.0+x*x))
    end
    pi_result = step_size*pi_sum[]
    return pi_result
end

@time @show thread_pi()
