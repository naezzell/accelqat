using DifferentialEquations
using CUDA
using LinearAlgebra

function cpu_example(n)
    tf = 10
    u0 = float_to_32_cpu(complex(ones(2^n)))
    u0 = u0 / norm(u0)
    # generate random 2^n x 2^n positive matrix (with complex entires)
    a0 = float_to_32_cpu(rand(2^n, 2^n) + 1.0im * rand(2^n, 2^n))
    H0 = conj(transpose(a0)) * a0
    a1 = float_to_32(rand(2^n, 2^n) + 1.0im * rand(2^n, 2^n))
    H1 = conj(transpose(a0)) * a0

    function interpolate_H(t)
        """
        Returns (1-s)*H0 + s*H1 where s ∈ [0, 1].
        """
        s = t / tf
        return (1-s)*H0 + s*H1
    end

    function update_func!(A, u, p, t)
        """
        Updator for DiffEqArrayOperator
        """
        fill!(A, 0.0)
        current_H = interpolate_H(t)
        @inbounds axpy!(-1.0im, current_H, A)
    end

    diff_op = DiffEqArrayOperator(-1.0im*copy(H0), update_func = update_func!)
    @show typeof(diff_op)
    jac_cache = similar(H0)
    jac_op = DiffEqArrayOperator(jac_cache, update_func = update_func!)
    ff = ODEFunction(diff_op, jac_prototype = jac_op)

    prob = ODEProblem{true}(ff, u0, (0.0, tf))
    sol = solve(prob; alg_hints = [:nonstiff])

    return sol
end


function gpu_example(n)
    tf = 10
    u0 = float_to_32_gpu(complex(ones(2^n)))
    u0 = cu(u0 / norm(u0))
    # generate random 2^n x 2^n positive matrix (with complex entires)
    a0 = float_to_32_gpu(rand(2^n, 2^n) + 1.0im * rand(2^n, 2^n))
    H0 = cu(conj(transpose(a0)) * a0)
    a1 = float_to_32_gpu(rand(2^n, 2^n) + 1.0im * rand(2^n, 2^n))
    H1 = cu(conj(transpose(a0)) * a0)

    function interpolate_H(t)
        """
        Returns (1-s)*H0 + s*H1 where s ∈ [0, 1].
        """
        s = t / tf
        return (1-s)*H0 + s*H1
    end

    function update_func!(A, u, p, t)
        """
        Updator for DiffEqArrayOperator
        """
        fill!(A, 0.0)
        current_H = interpolate_H(t)
        @inbounds axpy!(-1.0im, current_H, A)
    end

    diff_op = cu(DiffEqArrayOperator(-1.0im*copy(H0), update_func = update_func!))
    jac_cache = similar(H0)
    jac_op = cu(DiffEqArrayOperator(jac_cache, update_func = update_func!))
    ff = ODEFunction(diff_op, jac_prototype = jac_op)

    prob = ODEProblem{true}(ff, u0, (0.0, tf))
    sol = solve(prob; alg_hints = [:nonstiff])

    return sol
end
