using BenchmarkTools
using OpSum
using OpSum: mpo_bond_optimizations, BipartiteAlgorithm, SVDBondAlgorithm
using OpSum.PauliOperators: X, Y, Z

SUITE = BenchmarkGroup()

# ── Builder functions ──────────────────────────────────────────────────────────

function build_heisenberg(L)
    return opsum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1))
end

function build_j1j2(L; J1 = 1.0, J2 = 0.5)
    return J1 * opsum(X[i] * X[i + 1] for i in 1:(L - 1)) +
        J2 * opsum(X[i] * X[i + 2] for i in 1:(L - 2))
end

function build_all_to_all_xx(L)
    return opsum(sin(i + j) * X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L)
end

function build_haldane_shastry(N; J = 1.0)
    prefactor = J * π^2 / N^2
    return opsum(
        prefactor / sin(π * (m - n) / N)^2 * (X[n] * X[m] + Y[n] * Y[m] + Z[n] * Z[m])
            for n in 1:N for m in (n + 1):N
    )
end

# ── Benchmark registration ─────────────────────────────────────────────────────

# Dense all-to-all models: SVD has large bond matrices, so use smaller sizes.
let g = addgroup!(SUITE, "haldane_shastry"),
        g_c = addgroup!(g, "construction"),
        g_o = addgroup!(g, "optimization_bipartite"),
        g_s = addgroup!(g, "optimization_svd")

    for N in vcat(10:10:40, 50:25:300)
        H = build_haldane_shastry(N)
        g_c["N=$N"] = @benchmarkable build_haldane_shastry($N)
        g_o["N=$N"] = @benchmarkable mpo_bond_optimizations($(1:N), $H, $(BipartiteAlgorithm()))
    end
    for N in vcat(10:10:40, 50:25:100)
        H = build_haldane_shastry(N)
        g_s["N=$N"] = @benchmarkable mpo_bond_optimizations($(1:N), $H, $(SVDBondAlgorithm()))
    end
end

let g = addgroup!(SUITE, "all_to_all_xx"),
        g_c = addgroup!(g, "construction"),
        g_o = addgroup!(g, "optimization_bipartite"),
        g_s = addgroup!(g, "optimization_svd")

    for L in vcat(10:10:40, 50:25:300)
        H = build_all_to_all_xx(L)
        g_c["L=$L"] = @benchmarkable build_all_to_all_xx($L)
        g_o["L=$L"] = @benchmarkable mpo_bond_optimizations($(1:L), $H, $(BipartiteAlgorithm()))
    end
    for L in vcat(10:10:40, 50:25:100)
        H = build_all_to_all_xx(L)
        g_s["L=$L"] = @benchmarkable mpo_bond_optimizations($(1:L), $H, $(SVDBondAlgorithm()))
    end
end

# Short-range models: SVD stays cheap, use same sizes as bipartite.
let g = addgroup!(SUITE, "heisenberg"),
        g_c = addgroup!(g, "construction"),
        g_o = addgroup!(g, "optimization_bipartite"),
        g_s = addgroup!(g, "optimization_svd")

    for L in vcat(10:10:40, 50:25:500)
        H = build_heisenberg(L)
        g_c["L=$L"] = @benchmarkable build_heisenberg($L)
        g_o["L=$L"] = @benchmarkable mpo_bond_optimizations($(1:L), $H, $(BipartiteAlgorithm()))
        g_s["L=$L"] = @benchmarkable mpo_bond_optimizations($(1:L), $H, $(SVDBondAlgorithm()))
    end
end

let g = addgroup!(SUITE, "j1j2"),
        g_c = addgroup!(g, "construction"),
        g_o = addgroup!(g, "optimization_bipartite"),
        g_s = addgroup!(g, "optimization_svd")

    for L in vcat(10:10:40, 50:25:500)
        H = build_j1j2(L)
        g_c["L=$L"] = @benchmarkable build_j1j2($L)
        g_o["L=$L"] = @benchmarkable mpo_bond_optimizations($(1:L), $H, $(BipartiteAlgorithm()))
        g_s["L=$L"] = @benchmarkable mpo_bond_optimizations($(1:L), $H, $(SVDBondAlgorithm()))
    end
end
