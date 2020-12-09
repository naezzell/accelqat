using QuantumAnnealingTools, CUDA, DifferentialEquations
import QuantumAnnealingTools.⊗

function make_connect_tuples(n::Int64)
    """
    Make a list of all possible 2-body interactions for an
    [n] qubit system excluding duplicates, i.e. (i, j) where
    i < j and not both (i, j) and (j, i).

    Ex: [1, 2, 3] --> [(1, 2), (1, 3), (2, 3)]
    """
    connections = Array{Any,1}([])
    for prod in Iterators.product(1:n, 1:n)
        if (prod[1] < prod[2])
            push!(connections, prod)
        end
    end
    return connections
end


function make_spin_glass(n::Int64)
    """
    Create an [n] qubit spin-glass with couplings sampled
    (on the fly) from N(0, 1) distribution.
    """
    connections = make_connect_tuples(n)
    coeffs = randn(length(connections))
    return two_local_term(coeffs, connections, n)
end

function anneal_spin_glass(n::Int64, tf::T where T<:Number)
    """
    Performs a standard quantum anneal on an [n] qubit spin-glass for [tf]
    nano-seconds. Obtains spin-glass couplings from [load_dir].
    """
    # make Hamiltonians
    init_H = float_to_32_cpu(standard_driver(n))
    final_H = float_to_32_cpu(make_spin_glass(n))
    # get initial state
    init_state = ⊗([PauliVec[1][2] for _ in 1:n]...)
    H = DenseHamiltonian([(s)->1-s, (s)->s], [init_H, final_H], unit=:ħ)
    annealing = Annealing(H, init_state)
    sol = solve_schrodinger(annealing, tf)

    return sol
end

function anneal_spin_glass_gpu(n::Int64, tf::T where T<:Number)
    """
    Performs a standard quantum anneal on an [n] qubit spin-glass for [tf]
    nano-seconds. Obtains spin-glass couplings from [load_dir].
    """
    # make Hamiltonians
    init_H = float_to_32_gpu(cu(standard_driver(n)))
    final_H = float_to_32_gpu(cu(make_spin_glass(n)))
    # get initial state
    init_state = cu(⊗([PauliVec[1][2] for _ in 1:n]...))
    H = DenseHamiltonian([(s)->1-s, (s)->s], [init_H, final_H], unit=:ħ)
    annealing = Annealing(H, init_state)
    sol = solve_schrodinger_gpu(annealing, tf)

    return sol
end

function float_to_32_cpu(n::Array{Complex{Float64}})::Array{Complex{Float32}}
    """
    Convert to Float 64 arrays to Float 32 for improved runtime efficiency
    """
    return convert(Array{Complex{Float32}},n)
end

function float_to_32_gpu(n::CuArray{Complex{Float64}})::CuArray{Complex{Float32}}
    """
    Convert to Float 64 CUDA arrays to Float 32 for improved runtime efficiency
    """
    return convert(CuArray{Complex{Float32}},n)
end

@time anneal_spin_glass(3, 10);
@time anneal_spin_glass_gpu(3, 10);
