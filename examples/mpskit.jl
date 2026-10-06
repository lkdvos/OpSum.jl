# # Handing the MPO to MPSKit
#
# OpSum compresses an operator; it does not solve anything with it. This page is the seam: how a
# compressed MPO becomes an MPSKit `FiniteMPOHamiltonian`, and how to check that what comes out the
# other side is the operator you wrote down.
#
# There are two emissions, and picking the right one is the whole story:
#
# | function | tensors | what consumes it |
# |---|---|---|
# | [`irrep_mpo_tensors`](@ref OpSum.irrep_mpo_tensors) | one `TensorMap` per site | contraction, dense comparison, `MPSKit.InfiniteMPO` |
# | [`jordan_mpo_tensors`](@ref OpSum.jordan_mpo_tensors) | one `SparseBlockTensorMap` per site | `MPSKit.FiniteMPOHamiltonian`, and hence DMRG |
#
# `jordan_mpo_tensors` is the one MPSKit's Hamiltonian machinery wants: it reorders the bond indices
# so the start channel is first and the finish channel last, pads both identity channels where the
# cover did not spend an index on them, and emits diagonal pass-throughs as `BraidingTensor`s. That
# shape is what lets MPSKit keep identity chains out of dense storage — and, for a fermionic bond
# charge crossing a site, is what carries the sign.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

using MPSKit: MPSKit, JordanMPOTensor, FiniteMPOHamiltonian, FiniteMPS, DMRG, find_groundstate,
    expectation_value
using BlockTensorKit: SparseBlockTensorMap, nonzero_pairs, nonzero_keys

V = SU2Space(1 // 2 => 1)
S = spin(V)
chain(N) = FiniteChain(V, N)

# ## The Hamiltonian
#
# Nothing new — an SU(2) Heisenberg chain, the same operator as on the spin-chains page.

N = 6
H = opsum(chain(N), dot(S[i], S[i + 1]) for i in 1:(N - 1))
Ws = jordan_mpo_tensors(H)
map(W -> size(W, 4), Ws)

# Those are the Jordan-form bond sizes: `3` in the bulk for a Heisenberg chain, closing to `1` at
# the right edge. Two of the three bulk indices are the identity channels, which is why a
# nearest-neighbour model in Jordan form is `+2` over the unconstrained minimum that
# [`irrep_mpo`](@ref OpSum.irrep_mpo) reports.

# ## The one wart: boundary tensors
#
# `JordanMPOTensor(::SparseBlockTensorMap)` is the constructor this emission targets, and in the
# *bulk* it takes the tensors exactly as they come:

J = JordanMPOTensor(Ws[3])
Set(nonzero_keys(J)) == Set(nonzero_keys(Ws[3]))

# At the boundaries it throws instead. That is not a bug in either package: the constructor asserts
# that both diagonal corners are identities, and a boundary tensor cannot satisfy it — site 1 has a
# single row, so its `(end, end)` corner is the `(1, end)` slot, which holds an on-site term or
# nothing at all.
#
# So route the tensors through the same `undef` + `setindex!` path the constructor itself uses,
# uniformly for every site. Uniformly matters: mixing the two spellings gives an abstractly-typed
# `Vector` of site tensors, and MPSKit's algorithms then run untyped.

function to_jordan(W)
    O = MPSKit.jordanmpotensortype(spacetype(W), storagetype(W))(undef, space(W))
    for (I, v) in nonzero_pairs(W)
        O[I] = v
    end
    return O
end

Hmpo = FiniteMPOHamiltonian(map(to_jordan, Ws))

# ## Checking the round trip
#
# MPSKit splits each site tensor into Jordan blocks (`A`/`B`/`C`/`D`), and those accessors *silently
# drop* an entry that sits outside the Jordan pattern. So the sharp check is site by site, not a
# contraction: a misplaced entry shows up here as a missing block.

all(SparseBlockTensorMap(J) ≈ W for (W, J) in zip(Ws, parent(Hmpo)))

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

# ## What the infinite path can and cannot do
#
# On an `InfiniteChain`, `irrep_mpo` returns an `InfiniteMPO` whose tensors tile: the left virtual
# space of the first equals the right virtual space of the last.

lat∞ = InfiniteChain([V])
H∞ = opsum(lat∞, dot(S[1], S[2]))               # a generating set: one term per translation class
T∞ = irrep_mpo_tensors(irrep_mpo(H∞), lat∞)
space(T∞[1], 1) == space(T∞[end], 4)'

# That tiling is exactly `MPSKit.InfiniteMPO`'s requirement, so the plain MPO hands over:

MPSKit.InfiniteMPO(T∞)

# An infinite *Hamiltonian* does not, and it is worth being plain about why rather than letting a
# user find out by hitting the throw. `MPSKit.InfiniteMPOHamiltonian` requires `JordanMPOTensor`
# elements, and `jordan_mpo_tensors` is defined on a finite chain only:

try
    jordan_mpo_tensors(H∞)
catch e
    println(sprint(showerror, e))
end

# So VUMPS/IDMRG on an OpSum-built infinite Hamiltonian needs infinite Jordan emission, which does
# not exist yet. The finite path above is complete; the infinite path stops at `InfiniteMPO`.

# ## Name clashes
#
# `using OpSum, MPSKit` leaves `FiniteMPO` and `InfiniteMPO` ambiguous — both packages export both
# names, for different types (OpSum's hold `Ws` + `bondsectors`; MPSKit's hold site tensors). Qualify
# them, as this page does with `MPSKit.InfiniteMPO`.
