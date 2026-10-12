using SparseArrays: SparseMatrixCSC
using TensorKit: Sector, dim, unit

"""
    SymbolicMPO{I<:Sector, L<:AbstractLattice}

A compressed MPO with symbolic entries, as [`mpo`](@ref) returns it: `W[i]` is the sparse bond matrix of
site `i` (entries are [`SiteOperator`](@ref)s) and `W.bondsectors[i]` the charge of each index of the bond
to its right (on an [`InfiniteChain`](@ref) bond `L` is the one left of site 1). `W.start[i]` and
`W.finish[i]` are the two identity-channel indices on bond `i` (`0` if the compression spent none; both
vectors are empty when unknown, i.e. decaying channels on a finite chain).

`instantiate(W)` gives MPSKit's `FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian` once MPSKit is loaded.
"""
struct SymbolicMPO{I <: Sector, L <: AbstractLattice}
    lattice::L
    Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}
    bondsectors::Vector{Vector{I}}
    start::Vector{Int}
    finish::Vector{Int}
end

Base.length(W::SymbolicMPO) = length(W.Ws)
Base.getindex(W::SymbolicMPO, i::Integer) = W.Ws[i]
Base.iterate(W::SymbolicMPO, args...) = iterate(W.Ws, args...)
Base.eltype(::Type{<:SymbolicMPO{I}}) where {I} = SparseMatrixCSC{SiteOperator{I}, Int}
Base.firstindex(::SymbolicMPO) = 1
Base.lastindex(W::SymbolicMPO) = length(W)

function Base.show(io::IO, W::SymbolicMPO{I, L}) where {I, L}
    D = [sum(dim, sec; init = 0) for sec in W.bondsectors]
    return print(io, "SymbolicMPO{", I, "}(", nameof(L), ", N = ", length(W), ", D = ", D, ")")
end

"""
    mpo(H::OperatorSum[, alg]) -> SymbolicMPO

Compress `H` into a [`SymbolicMPO`](@ref), on either lattice, with or without exponentially decaying
channels ([`expterm`](@ref)).

On a `FiniteChain` the operator is `H` as written. On an `InfiniteChain` of `L` sites `H` is a generating
set: it represents `Σ_{n ∈ ℤ} translate(H, n·L)`, so each translation class must appear once (see
`unitcell_terms`). A decaying channel becomes one bond index with `λ` on its diagonal; on a
`FiniteChain` it represents the geometric sum truncated to the chain ([`chain_terms`](@ref)).

The two identity channels of every bond are identified (the Jordan form needs them); on a finite chain the
finish channel is forced into the cover for that, except with decaying channels.

`alg` is `BipartiteAlgorithm()` (default, lossless minimum vertex cover) or `SVDBondAlgorithm(trunc; sweep)`
(finite chains without channels only).
"""
function mpo(
        H::OperatorSum{I, <:FiniteChain},
        alg::Union{BipartiteAlgorithm, SVDBondAlgorithm} = BipartiteAlgorithm()
    ) where {I}
    return SymbolicMPO(H.lattice, _compress(H, alg)...)
end

mpo(H::OperatorSum{I, <:InfiniteChain}, ::BipartiteAlgorithm = BipartiteAlgorithm()) where {I} =
    _infinite_window(unitcell_terms(H))

mpo(::OperatorSum{I, <:InfiniteChain}, ::SVDBondAlgorithm) where {I} = throw(
    ArgumentError(
        "SVDBondAlgorithm does not extend to infinite chains: it compresses each bond " *
            "independently against a vacuum-terminated layout, with no bond basis that closes " *
            "on itself. Use BipartiteAlgorithm() (the default)."
    )
)

mpo(::Union{Term, Terms, ExpSum}, args...) = throw(
    ArgumentError("mpo takes an OperatorSum: put the terms on a lattice first, `mpo(opsum(lat, h))`")
)

SymbolicMPO(lat::L, Ws, secs::Vector{Vector{I}}, start, finish) where {I, L} =
    SymbolicMPO{I, L}(lat, Ws, secs, start, finish)

mpo_terms(W::SymbolicMPO{I, <:FiniteChain}) where {I} = mpo_terms(W.Ws, W.bondsectors)

irrep_mpo_tensors(W::SymbolicMPO{I, <:FiniteChain}) where {I} =
    irrep_mpo_tensors(W.Ws, W.bondsectors, W.lattice)
# the unit cell: as the finite assembly, but the bond left of site 1 is bond `L`, so the tensors tile
function irrep_mpo_tensors(W::SymbolicMPO{I, <:InfiniteChain}) where {I}
    L = length(W)
    bsecs = Vector{I}[W.bondsectors[mod1(j - 1, L)] for j in 1:(L + 1)]
    return _mpo_tensors(W.Ws, bsecs, W.lattice)
end

function jordan_mpo_tensors(W::SymbolicMPO{I, <:FiniteChain}) where {I}
    isempty(W.Ws) && throw(
        ArgumentError("cannot emit a Jordan MPO for an empty operator: it has no bond structure")
    )
    length(W.start) == length(W) || throw(
        ArgumentError(
            "this SymbolicMPO does not know its identity channels (it has exponentially decaying " *
                "channels on a finite chain), so it has no Jordan form yet"
        )
    )
    N = length(W)
    W.bondsectors[N] == I[unit(I)] || throw(
        ArgumentError(
            "the right boundary bond is $(W.bondsectors[N]), not the trivial `[$(unit(I))]`: only a " *
                "total-charge-`unit` operator has a Jordan MPO"
        )
    )
    # the boundary bonds stay one-dimensional
    bonds = [_jordan_bond(W.bondsectors[i], W.start[i], W.finish[i]) for i in 1:(N - 1)]
    pos = Vector{Int}[Int[1], first.(bonds)..., Int[1]]
    jsec = Vector{I}[I[unit(I)], last.(bonds)..., I[unit(I)]]
    return _jordan_tensors(W.Ws, pos, jsec, W.lattice)
end

# The unit cell: every bond carries both identity channels; the bond left of site 1 is bond `L`.
function jordan_mpo_tensors(W::SymbolicMPO{I, <:InfiniteChain}) where {I}
    L = length(W)
    bonds = [_jordan_bond(W.bondsectors[j], W.start[j], W.finish[j]) for j in 1:L]
    pos = Vector{Int}[first(bonds[mod1(j - 1, L)]) for j in 1:(L + 1)]
    jsec = Vector{I}[last(bonds[mod1(j - 1, L)]) for j in 1:(L + 1)]
    return _jordan_tensors(W.Ws, pos, jsec, W.lattice)
end

"""
    instantiate(W::SymbolicMPO)

MPSKit's `FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian` for `W`, built from its Jordan form.
Requires MPSKit to be loaded.
"""
instantiate(W::SymbolicMPO) = _mpskit_hamiltonian(W)

_mpskit_hamiltonian(::Any) = throw(
    ArgumentError("instantiate(::SymbolicMPO) builds an MPSKit Hamiltonian: load MPSKit first (`using MPSKit`)")
)
