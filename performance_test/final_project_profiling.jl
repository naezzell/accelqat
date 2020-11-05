using QuantumAnnealingTools
using DifferentialEquations
using BenchmarkTools
using JLD2
using Profile
using Plots

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

function gen_and_save_spin_glass_coeffs(n::Int64, dir::String)
    """
    Create a list of spin-glass couplings sampled from an
    N(0, 1) distribution. Then saves them to [dir].
    """
    connections = make_connect_tuples(n)
    coeffs = randn(length(connections))
    filename = "$(dir)$(n)_qubit_spin_glass_coeffs"
    @save filename coeffs
end

function load_spin_glass_coeffs(n::Int64, dir::String)
    """
    Loads in spin-glass couplings located in [dir] for
    an n-qubit system.
    """
    filename = "$(dir)$(n)_qubit_spin_glass_coeffs"
    @load filename coeffs
    return coeffs
end

function make_spin_glass(n::Int64, load_dir::String)
    """
    Creates an [n] qubit spin-glass with couplings loaded
    in from [load_dir].
    """
    connections = make_connect_tuples(n)
    coeffs = load_spin_glass_coeffs(n, load_dir)
    return two_local_term(coeffs, connections, n)
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

function anneal_spin_glass(n::Int64, tf::T where T<:Number, load_dir::String)
    """
    Performs a standard quantum anneal on an [n] qubit spin-glass for [tf]
    nano-seconds. Obtains spin-glass couplings from [load_dir].
    """
    # make Hamiltonians
    init_H = standard_driver(n)
    final_H = make_spin_glass(n, load_dir)
    # get initial state
    init_state = ⊗([PauliVec[1][2] for _ in 1:3]...)
    H = DenseHamiltonian([(s)->1-s, (s)->s], [init_H, final_H], unit=:ħ)
    annealing = Annealing(H, init_state)
    sol = solve_schrodinger(annealing, tf)

    return sol
end

function anneal_spin_glass(n::Int64, tf::T where T<:Number)
    """
    Performs a standard quantum anneal on an [n] qubit spin-glass for [tf]
    nano-seconds. Obtains spin-glass couplings from [load_dir].
    """
    # make Hamiltonians
    init_H = standard_driver(n)
    final_H = make_spin_glass(n)
    # get initial state
    init_state = ⊗([PauliVec[1][2] for _ in 1:3]...)
    H = DenseHamiltonian([(s)->1-s, (s)->s], [init_H, final_H], unit=:ħ)
    annealing = Annealing(H, init_state)
    sol = solve_schrodinger(annealing, tf)

    return sol
end

# basic timing function (builtin)
@time anneal_spin_glass(3, 20)

# bechmarking function (runs timing many times for statistics)
@benchmark anneal_spin_glass(3, 20)

# invoking the built-in statistical profiler
@profile anneal_spin_glass(3, 20)
# printing the results of priler to REPL
Profile.print()

# reset the profiler
Profile.clear()

# use the JUNO IDE proiler with the bar visualization
@profiler anneal_spin_glass(3, 20)

# use the JUNO IDE profiler with combine = true
@profiler anneal_spin_glass(3, 20) combine = true


# just a bonus to see the distribution of spin-glass couplings shown graphically
histogram(randn(10000))
