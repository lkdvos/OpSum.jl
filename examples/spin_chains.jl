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

using OpSum: arity

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
# Note the two idioms used here and throughout. `S = spin(V)` is hoisted out of the loop (it is
# memoised, so this is taste rather than necessity), and the terms are handed to
# [`opsum`](@ref OpSum.opsum) in one pass — it is the linear accumulator, where folding `+` would
# copy on every step.
#
# `opsum(lat, terms...)` returns an [`OperatorSum`](@ref OpSum.OperatorSum): the terms together with
# their lattice. The term algebra itself (`dot`, `couple`, `A[i]`) is **latticeless** — it needs no
# physical space, and the compression needs only the number of sites — so the lattice enters here,
# when the terms are added, which is also where every letter is checked against the space of the
# site it sits on. [`irrep_mpo`](@ref OpSum.irrep_mpo), `instantiate`, `islossless` and `H'` all take
# the `OperatorSum`.

V = SU2Space(1 // 2 => 1)
S = spin(V)

su2chain(N) = FiniteChain(V, N)          # `FiniteChain(V, N)` replaces `fill(V, N)`
heisenberg(N; J = 1.0) = opsum(su2chain(N), J * dot(S[i], S[i + 1]) for i in 1:(N - 1))

N = 8
H_heis = heisenberg(N)
res_heis = build("Heisenberg SU(2)", H_heis)

# The compression is lossless — the reduced MPO reconstructs every original term exactly:

islossless(H_heis)

# and contracting the assembled tensors reproduces the dense operator:

mpo_matches_oracle(heisenberg(4))

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
# than hard-coding alphabet indices, `spin_ops` derives them from the matrix units, which is robust
# against how `V` happens to be written down. It takes the sectors in *descending* ``m`` — the labels
# themselves cannot say which is which, since a `Vect[U₁]` spin site is as often labelled by particle
# number as by ``m``, exactly as here.

Vu = Rep[U₁](0 => 1, 1 => 1)
dn, up = U1Irrep(0), U1Irrep(1)

Sp, Sm, Sz = spin_ops(Vu, up, dn)                        # ``S^+``, ``S^-``, ``S^z``

# ``S^z`` is a genuinely *composite* on-site operator — a two-letter combination:

length(Sz)

# `couple` distributes over both expansions, so a composite operand needs no special handling: the
# ``S^z S^z`` term is written exactly like the single-letter ones.

u1chain(N) = FiniteChain(Vu, N)
function xxz(N; J = 1.0, Δ = 1.0)
    return opsum(
        u1chain(N),
        J / 2 * couple(Sp[i], Sm[i + 1]) +
            J / 2 * couple(Sm[i], Sp[i + 1]) +
            J * Δ * couple(Sz[i], Sz[i + 1])
            for i in 1:(N - 1)
    )
end

H_xxz = xxz(N)
res_xxz = build("XXZ U(1)", H_xxz)

islossless(H_xxz)

# At ``\Delta = 1`` the XXZ chain *is* the Heisenberg chain. The two builds live on different
# spaces with different symmetry groups, so the sharpest available check is that they have the
# same spectrum:

spectrum(heisenberg(6)) ≈ spectrum(xxz(6))

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

# ## Projecting a whole bond at once
#
# Naming the on-site factors is not the only option. If you already have the two-site bond operator
# as a `TensorMap` — from a paper's matrix, from an ED code, from `kron` — hand it straight to
# `project`, which expands it in the two-site ITO term basis. Nothing has to factorize into
# ``A_i B_j``: a generic two-site block works, and the coefficients are exact inner products
# against a complete orthogonal basis, so the expansion is not a fit.

h_bond = instantiate(
    opsum(
        [Vu, Vu],
        (1 / 2) * couple(Sp[1], Sm[2]) +
            (1 / 2) * couple(Sm[1], Sp[2]) +
            couple(Sz[1], Sz[2]),
    )
)

B_xxz = project(h_bond)

# The result is a [`LocalOperator`](@ref OpSum.LocalOperator): the two-site operator *before* it is
# told where it acts. `B_xxz[i]` places it on sites `i, i + 1`, so the projection — and its
# faithfulness check, which re-materializes the output and compares it against the input — is paid
# once rather than once per bond:

H_proj = opsum(u1chain(N), B_xxz[i] for i in 1:(N - 1))

# Summed over bonds it reproduces the hand-written chain, term for term:

H_proj ≈ H_xxz

# The one thing to know: every projected term is active on *all* `K` slots. An on-site identity
# factor comes back as a trivial-charge letter rather than a shorter term, so `project` inverts
# `instantiate` only for operators whose terms have full support on the block.

# ## There is no symbolic on-site product
#
# The bilinear-biquadratic spin-1 chain is the sharpest case for `project`, and the motivation comes
# from the physics rather than from the API:
#
# ```math
# H = \sum_i \left[ \vec{S}_i\!\cdot\!\vec{S}_{i+1} + \beta \left( \vec{S}_i\!\cdot\!\vec{S}_{i+1} \right)^2 \right]
# ```
#
# The square is a *product of two-site operators*, and OpSum has no lazy symbolic algebra to
# multiply them in: `Terms` is a flat bag of terms, not an expression tree. So the way to square
# something is to build the block, square *that*, and project the result.

V1 = SU2Space(1 => 1)
S1 = spin(V1)
bond = removeunit(instantiate(opsum([V1, V1], dot(S1[1], S1[2]))), 5)

# `instantiate` returns the operator with a trailing trivial charge leg, and `removeunit` drops it,
# leaving the ordinary ``V \otimes V \leftarrow V \otimes V`` map that can be multiplied:

space(bond)

# At ``\beta = 1/3`` this is the AKLT chain, whose ground state is the valence-bond solid:

function bilinear_biquadratic(N; β = 1 / 3)
    B = project(bond + β * (bond * bond))
    return opsum(FiniteChain(V1, N), B[i] for i in 1:(N - 1))
end

H_aklt = bilinear_biquadratic(6)
build("AKLT (β=1/3)", H_aklt)
islossless(H_aklt)

# The biquadratic term costs bond dimension because a spin-2 channel opens alongside the spin-1 one:

for β in (0.0, 1 / 3, 1.0)
    r = build("β=$(round(β; digits = 3))", bilinear_biquadratic(6; β); quiet = true)
    println("  β=$(rpad(round(β; digits = 3), 5))  D=$(r.D)  D_dense=$(r.Ddense)")
end

# The ``\beta = 0`` row is the plain spin-1 Heisenberg chain, and here the round trip is exact as a
# *term bag*, not merely as an operator — `project` recovers the very terms `dot` would have
# produced:

let plain = opsum(FiniteChain(V1, 6), dot(S1[i], S1[i + 1]) for i in 1:5)
    (; same_bag = bilinear_biquadratic(6; β = 0.0) ≈ plain, nterms = length(plain))
end

# That is worth separating from the full-support property, because the two are easy to conflate. A
# projected term is active on *all* `K` slots of the block, and the reason it costs nothing above is
# that both terms of a `dot` already span both sites. Add a piece that does not — an identity, say —
# and it comes back padded, as a two-slot term carrying a trivial-charge letter rather than as a
# shorter term:

let padded = project(bond + one(bond) / 4)
    (; nterms = length(padded), arities = unique(arity(t) for t in padded))
end

# Both terms have arity 2: the identity did not come back as a `K = 0` term. So `project` inverts
# `instantiate` for operators whose terms have full support on the block, and pads everything else —
# which is exactly what makes it a faithful expansion of a *block* rather than a factorization.

# ## ``J_1``–``J_2``
#
# ```math
# H = J_1 \sum_i \vec{S}_i \cdot \vec{S}_{i+1} + J_2 \sum_i \vec{S}_i \cdot \vec{S}_{i+2}
# ```
#
# Adding a next-nearest-neighbour coupling means two spin-1 channels can be open across a bond at
# once, so the bond dimension grows — but it is still independent of ``N``.

function j1j2(N; J1 = 1.0, J2 = 0.5)
    return opsum(
        su2chain(N),
        (J1 * dot(S[i], S[i + 1]) for i in 1:(N - 1)),
        (J2 * dot(S[i], S[i + 2]) for i in 1:(N - 2)),
    )
end

H_j1j2 = j1j2(N)
res_j1j2 = build("J1-J2 SU(2)", H_j1j2)
islossless(H_j1j2)

# ## Bond dimension is independent of system size
#
# For any finite-range interaction the bond dimension saturates: it is set by how many couplings
# can straddle a single cut, not by how long the chain is.

for L in (8, 16, 32, 64)
    r = build("h", heisenberg(L); quiet = true)
    println("  Heisenberg  N=$(lpad(L, 3))  D=$(r.D)  D_dense=$(r.Ddense)")
end

for L in (8, 16, 32, 64)
    r = build("j", j1j2(L); quiet = true)
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
#
# The time panel above is the whole pipeline: symbolic term accumulation *plus* MPO compression.
# Splitting the two (`--figure phases`) shows where the work actually goes — for a finite-range model
# the compression is linear in ``N``, because each term is an open channel only on the bonds it
# actually straddles, so the total is dominated by the term accumulation:
#
# ![Term-sum assembly versus MPO compression](../assets/phases.png)
