# # Spin chains
#
# This page builds the three workhorse one-dimensional spin models and shows how the *symmetry* you
# impose and the *range* of the interaction each leave a distinct fingerprint on the MPO bond
# dimension:
#
# | model | symmetry | reduced ``D`` | dense-equivalent ``D`` |
# |:--|:--|--:|--:|
# | Heisenberg | SU(2) | 3 | 5 |
# | XXZ | U(1) | 6 | 6 |
# | ``J_1``–``J_2`` | SU(2) | 4 | 8 |
#
# The reduced bond dimension counts symmetry-resolved indices — what a DMRG sweep actually pays
# for. The dense-equivalent one is what a symmetry-agnostic MPO would need for the same operator.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## SU(2) Heisenberg
#
# ```math
# H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1}
# ```
#
# With SU(2) symmetry there is a single on-site operator to speak of: the rank-1 vector operator
# ``\vec{S}``, provided by [`spin`](@ref OpSum.spin). `dot` contracts two of them into the rotationally
# invariant scalar product, so the Hamiltonian is a one-liner.
#
# Note the two idioms used here and throughout. `S = spin(V)` is hoisted out of the loop — it
# recomputes its normalization on every call otherwise — and the terms are collected into a
# `Vector` before summing. Never write `reduce(+, generator)`: adding two `TermSum`s rebuilds the
# underlying dictionary, and folding over a generator makes that quadratic in the number of terms.

V = SU2Space(1 // 2 => 1)
S = spin(V)

heisenberg(N; J = 1.0) = sum([J * dot(S[i], S[i + 1]) for i in 1:(N - 1)])

N = 8
sites = fill(V, N)
H_heis = heisenberg(N)
res_heis = build("Heisenberg SU(2)", H_heis, sites)

# The compression is lossless — the reduced MPO reconstructs every original term exactly:

islossless(H_heis, sites)

# and contracting the assembled tensors reproduces the dense operator:

mpo_matches_oracle(heisenberg(4), fill(V, 4))

# The bulk bond carries an identity-in channel, an identity-out channel and one open spin-1
# multiplet: ``1 + 1 + 3 = 5`` in dense terms, but only 3 symmetry-resolved indices.

res_heis.D, res_heis.Ddense

# ## U(1) XXZ
#
# ```math
# H = \frac{J}{2} \sum_i \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right)
#     + J \Delta \sum_i S^z_i S^z_{i+1}
# ```
#
# Keeping only ``U(1)`` means naming individual raising, lowering and ``S^z`` operators. Rather
# than hard-coding alphabet indices we derive them from the matrix units (see
# [Shared utilities](@ref)), which is robust against how `V` happens to be written down.

Vu = Rep[U₁](0 => 1, 1 => 1)
dn, up = U1Irrep(0), U1Irrep(1)

Sp = matrixunit(Vu, up, dn)                              # ``S^+``
Sm = matrixunit(Vu, dn, up)                              # ``S^-``
Sz = (matrixunit(Vu, up, up) - matrixunit(Vu, dn, dn)) / 2 # ``S^z``

# ``S^z`` is a genuinely *composite* on-site operator — a two-letter combination:

length(Sz.terms)

# That matters, because `couple` builds its fusion tree one leg at a time and so accepts only
# single-letter operands: `couple(Sz[i], Sz[j])` throws. `bondterm` distributes the coupling over
# both expansions, which is what makes the ``S^z S^z`` term expressible.

function xxz(N; J = 1.0, Δ = 1.0)
    z = unit(U1Irrep)
    return sum(
        [
            J / 2 * bondterm(Sp, i, Sm, i + 1; to = z) +
                J / 2 * bondterm(Sm, i, Sp, i + 1; to = z) +
                J * Δ * bondterm(Sz, i, Sz, i + 1; to = z)
                for i in 1:(N - 1)
        ]
    )
end

sites_u1 = fill(Vu, N)
H_xxz = xxz(N)
res_xxz = build("XXZ U(1)", H_xxz, sites_u1)

islossless(H_xxz, sites_u1)

# At ``\Delta = 1`` the XXZ chain *is* the Heisenberg chain. The two builds live on different
# spaces with different symmetry groups, so the sharpest available check is that they have the
# same spectrum:

spectrum(heisenberg(6), fill(V, 6)) ≈ spectrum(xxz(6), fill(Vu, 6))

# The same operator, but not the same MPO. The SU(2) build needs 3 symmetry-resolved indices where
# the U(1) build needs 6, because a single spin-1 multiplet replaces three separate abelian channels
# (``+1``, ``-1``, ``0``). The reduced number is what a DMRG sweep pays for, so this factor of two
# is the practical benefit of imposing the larger symmetry.
#
# The dense-equivalent sizes differ slightly too (5 versus 6). That is a property of how each
# Hamiltonian is *written*: ``S^z`` is a two-letter operator, so ``S^z_i S^z_j`` enters as four
# letter pairs which do not collapse into a single channel, whereas the SU(2) scalar product is one
# irreducible object.

(su2 = (res_heis.D, res_heis.Ddense), u1 = (res_xxz.D, res_xxz.Ddense))

# ## ``J_1``–``J_2``
#
# ```math
# H = J_1 \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + J_2 \sum_i \vec{S}_i \cdot \vec{S}_{i+2}
# ```
#
# Adding a next-nearest-neighbour coupling means two spin-1 channels can be open across a bond at
# once, so the bond dimension grows — but it is still independent of ``N``.

function j1j2(N; J1 = 1.0, J2 = 0.5)
    return sum(
        vcat(
            [J1 * dot(S[i], S[i + 1]) for i in 1:(N - 1)],
            [J2 * dot(S[i], S[i + 2]) for i in 1:(N - 2)],
        )
    )
end

H_j1j2 = j1j2(N)
res_j1j2 = build("J1-J2 SU(2)", H_j1j2, sites)
islossless(H_j1j2, sites)

# ## Bond dimension is independent of system size
#
# For any finite-range interaction the bond dimension saturates: it is set by how many couplings
# can straddle a single cut, not by how long the chain is.

for L in (8, 16, 32, 64)
    r = build("h", heisenberg(L), fill(V, L); quiet = true)
    println("  Heisenberg  N=$(lpad(L, 3))  D=$(r.D)  D_dense=$(r.Ddense)")
end

for L in (8, 16, 32, 64)
    r = build("j", j1j2(L), fill(V, L); quiet = true)
    println("  J1-J2       N=$(lpad(L, 3))  D=$(r.D)  D_dense=$(r.Ddense)")
end

# ## Scaling
#
# Construction time and bond dimension across system size, for these and the other models, are
# collected by the benchmark harness in `benchmark/` and plotted by
# `scripts/plot_benchmarks.jl`. Reproduce the figure with
#
# ```
# julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full
# ```
#
# ![Bond dimension and construction time versus system size](../assets/scaling.png)
