# Phase 3 benchmark: Trie vs TermTable BipartiteAlgorithm MPO construction.
#
# Measures end-to-end construction time and allocations of the two front ends
# across scaling in term count and chain length N, including a long-range
# (all-to-all) case — the one most likely to expose the pointer-chasing/hashing
# overhead of the Trie path.
#
# Run with:  julia --project research/bench_termtable.jl

using OpSum
using OpSum: TermTable, BipartiteAlgorithm, Trie, build_trie!
using OpSum.PauliOperators: X, Y, Z
using LinearAlgebra: norm
using Printf

# Old path: build the Trie, then optimize.
function run_trie(vertices, H)
    Ws = mpo_bond_optimizations(vertices, H, BipartiteAlgorithm())
    return Ws
end

# New path: build the TermTable, then optimize.
function run_termtable(vertices, H)
    tt = TermTable(vertices, H)
    Ws = mpo_bond_optimizations(vertices, tt, BipartiteAlgorithm())
    return Ws
end

# Minimal timing: best of `reps` runs (after one warmup), plus allocation count.
function timeit(f, args...; reps = 3)
    f(args...)                                   # warmup / compile
    best = Inf
    allocs = 0.0
    for _ in 1:reps
        t0 = time_ns()
        f(args...)
        dt = (time_ns() - t0) / 1e9
        best = min(best, dt)
    end
    allocs = (@allocated f(args...)) / 1e6       # MB
    return best, allocs
end

function bench_case(name, vertices, H)
    N = length(vertices)
    to, ao = timeit(run_trie, vertices, H)
    tn, an = timeit(run_termtable, vertices, H)
    @printf("%-28s N=%-4d  Trie: %8.4f s / %8.2f MB   TermTable: %8.4f s / %8.2f MB   speedup %5.2fx\n",
        name, N, to, ao, tn, an, to / tn)
    return nothing
end

println("=== MPO construction: Trie vs TermTable (BipartiteAlgorithm) ===")

# Nearest-neighbour XX + field, scaling N.
for N in (10, 20, 40, 80)
    H = sum(X[i] * X[i + 1] for i in 1:(N - 1)) + sum(Z[i] for i in 1:N)
    bench_case("nn-XX+field", 1:N, H)
end

# Heisenberg, scaling N.
for N in (10, 20, 40, 80)
    H = sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(N - 1))
    bench_case("heisenberg", 1:N, H)
end

# All-to-all XX (long range) — O(N^2) terms. Pointer-chasing stress case.
for N in (10, 20, 30, 40)
    H = sum(X[i] * X[j] for i in 1:(N - 1) for j in (i + 1):N)
    bench_case("all-to-all-XX", 1:N, H)
end
