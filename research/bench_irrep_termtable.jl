# Phase 4 benchmark: trie vs flat (TermTable) ITO MPO construction.
#
# Compares end-to-end irrep_mpo (trie) vs irrep_mpo_flat across SU(2) Heisenberg
# and U(1) hopping chains scaling in N. Also splits the flat path into
# storage-construction vs bipartite-sweep so we can see where time goes.
#
# Run with:  julia --project research/bench_irrep_termtable.jl

using OpSum
using OpSum: irrep_mpo, irrep_mpo_flat, irrep_trie, ITOTermTable,
    _irrep_bipartite, _irrep_bipartite_flat, spin
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot
using Printf

function timeit(f, args...; reps = 3)
    f(args...)                                   # warmup / compile
    best = Inf
    for _ in 1:reps
        t0 = time_ns()
        f(args...)
        best = min(best, (time_ns() - t0) / 1.0e9)
    end
    allocs = (@allocated f(args...)) / 1.0e6       # MB
    return best, allocs
end

irrep_mpo_trie(H, sites) = _irrep_bipartite(irrep_trie(H, sites), length(sites))

function bench_case(name, H, sites)
    N = length(sites)
    tt, at = timeit(irrep_mpo_trie, H, sites)     # trie end-to-end
    tf, af = timeit(irrep_mpo_flat, H, sites)     # flat end-to-end
    # split flat into storage + sweep
    ts, _ = timeit(ITOTermTable, H, sites)
    tbl = ITOTermTable(H, sites)
    tsw, _ = timeit(t -> _irrep_bipartite_flat(t, N), tbl)
    @printf(
        "%-22s N=%-3d  trie: %8.4f s /%8.2f MB   flat: %8.4f s /%8.2f MB  (store %7.4f + sweep %7.4f)  speedup %5.2fx\n",
        name, N, tt, at, tf, af, ts, tsw, tt / tf
    )
    return nothing
end

su2 = SU2Space(1 // 2 => 1)
u1 = Rep[U₁](0 => 1, 1 => 1)
LO(x) = OpSum.LocalOp(x)
raise = LO(IrrepOperator(U1Irrep(1), 1))
lower = LO(IrrepOperator(U1Irrep(-1), 1))

println("=== ITO MPO construction: trie vs flat ===")

# SU(2) Heisenberg, nearest neighbour.
for N in (8, 16, 32, 48)
    H = reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:(N - 1))
    bench_case("su2-heisenberg-nn", H, fill(su2, N))
end

# SU(2) next-nearest-neighbour (J1-J2), denser bonds.
for N in (8, 16, 32)
    H = reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:(N - 1)) +
        reduce(+, dot(spin(su2)[i], spin(su2)[i + 2]) for i in 1:(N - 2))
    bench_case("su2-J1J2", H, fill(su2, N))
end

# U(1) hopping.
for N in (8, 16, 32)
    H = reduce(+, dot(raise[i], lower[i + 1]) for i in 1:(N - 1)) +
        reduce(+, dot(lower[i], raise[i + 1]) for i in 1:(N - 1))
    bench_case("u1-hopping", H, fill(u1, N))
end
