# # Handing the MPO to MPSKit
#
# OpSum compresses an operator; it does not solve anything with it. This page is the seam: how a
# compressed MPO becomes an MPSKit `FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian`, and how to check
# that what comes out the other side is the operator you wrote down.
#
# [`mpo`](@ref OpSum.mpo) compresses an `OperatorSum` into a [`SymbolicMPO`](@ref OpSum.SymbolicMPO),
# and with MPSKit loaded `instantiate(W)` turns that into MPSKit's Hamiltonian. Underneath, there are
# two tensor emissions:
#
# | function | tensors | what consumes it |
# |---|---|---|
# | [`irrep_mpo_tensors`](@ref OpSum.irrep_mpo_tensors) | one `TensorMap` per site | contraction, dense comparison, `MPSKit.InfiniteMPO` |
# | [`jordan_mpo_tensors`](@ref OpSum.jordan_mpo_tensors) | one `SparseBlockTensorMap` per site | `instantiate(W)`, and hence DMRG and VUMPS |
#
# `jordan_mpo_tensors` is the one MPSKit's Hamiltonian machinery wants: it reorders the bond indices
# so the start channel is first and the finish channel last, pads both identity channels where the
# cover did not spend an index on them, and emits diagonal pass-throughs as `BraidingTensor`s. That
# shape is what lets MPSKit keep identity chains out of dense storage — and, for a fermionic bond
# charge crossing a site, is what carries the sign.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

using MPSKit: MPSKit, JordanMPOTensor, FiniteMPOHamiltonian, InfiniteMPOHamiltonian, FiniteMPS,
    InfiniteMPS, DMRG, VUMPS, find_groundstate, expectation_value
using BlockTensorKit: SparseBlockTensorMap, nonzero_keys

B = spin_basis(SU2Irrep)
V = space(B)                      # the physical space of the MPS
ex = spin_ops(SU2Irrep).S_exchange
chain(N) = FiniteChain(B, N)

# ## The Hamiltonian
#
# Nothing new — an SU(2) Heisenberg chain, the same operator as on the spin-chains page.

N = 6
H = opsum(chain(N), ex[i, i + 1] for i in 1:(N - 1))
Ws = jordan_mpo_tensors(H)
map(W -> size(W, 4), Ws)

# Those are the Jordan-form bond sizes: `3` in the bulk for a Heisenberg chain, closing to `1` at
# the right edge. Two of the three bulk indices are the identity channels, which is why a
# nearest-neighbour model in Jordan form is `+2` over the minimum the cover itself reaches
# (`mpo(H).bondsectors`).

# ## Into MPSKit
#
# `mpo(H)` is the compressed operator, still symbolic: per site a sparse matrix of on-site operators,
# per bond its charges and its two identity channels. With MPSKit loaded, `instantiate` builds the
# Hamiltonian from its Jordan form:

W = mpo(H)
Hmpo = instantiate(W)

# MPSKit splits each site tensor into Jordan blocks (`A`/`B`/`C`/`D`), and those accessors *silently
# drop* an entry that sits outside the Jordan pattern. So the sharp check of the round trip is site by
# site, not a contraction: a misplaced entry shows up here as a missing block.

all(SparseBlockTensorMap(J) ≈ Wj for (Wj, J) in zip(jordan_mpo_tensors(W), parent(Hmpo)))

# The bulk tensors are valid `JordanMPOTensor`s as they come, too:

Set(nonzero_keys(JordanMPOTensor(Ws[3]))) == Set(nonzero_keys(Ws[3]))

# ## Ground state versus exact diagonalisation
#
# The physics check. `spectrum(H)` in `common.jl` diagonalises the dense oracle block by block,
# which is only possible because `N` is small — that is the point of checking here rather than
# trusting the pipeline.

exact = spectrum(H)[1]

# The MPS bond space needs half-integer *and* integer spins: an odd bond of a spin-½ chain carries
# the former.

ψ = FiniteMPS(randn, ComplexF64, N, V, SU2Space(0 => 8, 1 // 2 => 8, 1 => 8, 3 // 2 => 4))
ψ, = find_groundstate(ψ, Hmpo, DMRG(; tol = 1.0e-10, verbosity = 0))
E = real(expectation_value(ψ, Hmpo))
(; dmrg = E, exact, error = abs(E - exact))

# Spelled as a check, so that this page fails rather than quietly reporting a wrong number:

isapprox(E, exact; atol = 1.0e-8)

# ## The dense route, for comparison
#
# `irrep_mpo_tensors` gives plain `TensorMap`s, and [`mpo_tensormap`](@ref OpSum.mpo_tensormap)
# contracts them into `instantiate`'s convention. This is the check that the *operator* is right,
# independent of anything MPSKit does with it — and it is exponential in `N`, so it stays small.

mpo_matches_oracle(H)

# ## The infinite chain
#
# On an `InfiniteChain` the operator is a generating set, one term per translation class, and
# `instantiate(mpo(H∞))` is an `InfiniteMPOHamiltonian`: the unit cell's bond indices carry both
# identity channels, so its Jordan form needs no padding. A spin-½ chain under SU(2) needs a two-site
# cell, since the bond spaces alternate between half-integer and integer spins.

lat∞ = InfiniteChain([B, B])
H∞ = opsum(lat∞, ex[1, 2], ex[2, 3])
H∞mpo = instantiate(mpo(H∞))

# VUMPS then gives the Heisenberg ground-state energy per site, `1/4 - log(2)` (Bethe):

ψ∞ = InfiniteMPS([V, V], [SU2Space(1 // 2 => 10, 3 // 2 => 4), SU2Space(0 => 10, 1 => 8, 2 => 2)])
ψ∞, = find_groundstate(ψ∞, H∞mpo, VUMPS(; tol = 1.0e-8, verbosity = 0))
e∞ = real(expectation_value(ψ∞, H∞mpo)) / 2
(; vumps = e∞, bethe = 1 / 4 - log(2))

#-

isapprox(e∞, 1 / 4 - log(2); atol = 1.0e-3)

# The plain (non-Hamiltonian) tensors tile as well, which is what `MPSKit.InfiniteMPO` takes:

T∞ = irrep_mpo_tensors(mpo(H∞))
space(T∞[1], 1) == space(T∞[end], 4)'

# ## Name clashes
#
# `using OpSum, MPSKit` is safe: OpSum's MPO type is [`SymbolicMPO`](@ref OpSum.SymbolicMPO), so it
# does not collide with MPSKit's `FiniteMPO` and `InfiniteMPO`.
