# # Long-range interactions
#
# Every model on the previous pages had a bond dimension independent of system size, because only a
# bounded number of couplings could straddle any cut. Long-range models break that: when *every*
# pair of sites interacts, the number of open channels at a bond grows with the system.
#
# This page builds the Haldane–Shastry chain and a general power law, and shows that the exact bond
# dimension grows *linearly*:
#
# ```math
# D_\mathrm{dense} = \tfrac{3}{2} N + 2 .
# ```

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## Haldane–Shastry
#
# ```math
# H = J \frac{\pi^2}{N^2} \sum_{n < m} \frac{\vec{S}_n \cdot \vec{S}_m}{\sin^2\!\left(\pi (n-m)/N\right)}
# ```
#
# An all-to-all model on ``N`` sites has ``\binom{N}{2}`` terms, so the accumulation pattern matters
# a great deal here. Collecting the terms into a `Vector` and calling `sum` is ``O(M \log M)``;
# writing `reduce(+, generator)` instead folds left and rebuilds the term dictionary at every step,
# which is ``O(M^2)`` — at ``N = 120`` that is the difference between 0.75 s and 6.9 s.

V = SU2Space(1 // 2 => 1)
S = spin(V)

function haldane_shastry(N; J = 1.0)
    pref = J * π^2 / N^2
    return sum(
        [
            (pref / sin(π * (m - n) / N)^2) * dot(S[n], S[m])
                for n in 1:(N - 1) for m in (n + 1):N
        ]
    )
end

N = 16
sites = fill(V, N)
H_hs = haldane_shastry(N)
res_hs = build("Haldane-Shastry", H_hs, sites)

# Even with every pair coupled, the compression is still exact:

islossless(H_hs, sites)

# ## Linear growth
#
# The coefficient matrix of a generic long-range model has full rank, so at a cut after site ``b``
# the minimum vertex cover has to keep one open spin-1 channel for roughly every site on the
# smaller side — giving ``\min(b, N-b)`` multiplets, maximised at the middle of the chain.

for L in (10, 20, 40, 60, 80)
    r = build("HS N=$L", haldane_shastry(L), fill(V, L); quiet = true)
    println(
        "  N=$(rpad(L, 3))  nterms=$(rpad(L * (L - 1) ÷ 2, 5))  D=$(rpad(r.D, 4))",
        "  D_dense=$(rpad(r.Ddense, 5))  3N/2+2 = $(3L ÷ 2 + 2)"
    )
end

# Contrast that with the finite-range models: this is the one family whose MPO genuinely grows with
# the system, and it is why long-range Hamiltonians are the interesting stress test for MPO
# construction.

all(
    build("hs", haldane_shastry(L), fill(V, L); quiet = true).Ddense == 3L ÷ 2 + 2
        for L in (10, 20, 30, 40)
)

# ## A general power law
#
# ```math
# H = J \sum_{n<m} \frac{\vec{S}_n \cdot \vec{S}_m}{|n-m|^{\alpha}}
# ```
#
# The exponent controls how fast the couplings decay, but not the *exact* bond dimension: any
# coefficient matrix of full rank gives the same linear law. Truncation is what exploits the decay.

function powerlaw(N; α = 3.0, J = 1.0)
    return sum(
        [
            (J * abs(m - n)^(-α)) * dot(S[n], S[m])
                for n in 1:(N - 1) for m in (n + 1):N
        ]
    )
end

for α in (1.0, 2.0, 3.0, 6.0)
    r = build("powerlaw α=$α", powerlaw(24; α), fill(V, 24); quiet = true)
    println("  α=$(rpad(α, 4))  D=$(rpad(r.D, 4))  D_dense=$(r.Ddense)")
end

# ## Trading exactness for size
#
# `BipartiteAlgorithm` (the default) is exact. `SVDBondAlgorithm` instead compresses each bond by a
# truncated SVD across all charge sectors at once, so a fast-decaying power law can be squeezed hard
# for a controlled error.
#
# Note that faithfulness via `mpo_terms` is meaningless once the compression is lossy — it assumes
# an intact identity backbone. The honest check is the operator error against the dense oracle, so
# we measure that at a size where the oracle is affordable.

using MatrixAlgebraKit: truncrank

let sites_c = fill(V, 6), H = powerlaw(6; α = 3.0)
    oracle = instantiate(H, sites_c)
    exact = build("powerlaw exact", H, sites_c; quiet = true)
    println("  exact:            D_dense=$(exact.Ddense)   rel. error 0")
    for k in (8, 6, 4, 2, 1)
        Ws, secs = irrep_mpo(H, sites_c, SVDBondAlgorithm(truncrank(k)))
        O = mpo_tensormap(irrep_mpo_tensors(Ws, secs, sites_c))
        err = norm(O - oracle) / norm(oracle)
        D = maximum(b -> sum(dim(c) for c in secs[b]), eachindex(secs))
        println("  truncrank($(rpad(k, 2)))     D_dense=$(rpad(D, 4))  rel. error $(round(err; sigdigits = 3))")
    end
end

# ## Scaling
#
# ```
# julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full
# ```
#
# ![Bond dimension and construction time versus system size](../assets/scaling.png)
#
# The *profile* of the bond dimension along the chain makes the contrast with the local models
# vivid: long-range couplings give a triangle peaking at the middle of the chain, where the cut
# separates the largest number of interacting pairs, while finite-range and quasi-2D models sit on a
# flat plateau.
#
# ![Bond dimension profile along the chain](../assets/profile.png)
