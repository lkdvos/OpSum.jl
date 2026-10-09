# The symbolic MPO: what `mpo(H)` returns (operator-bases.md §3.8). Per site a sparse matrix of
# `SiteOperator`s — coefficient vectors over the site's basis — and per bond its charges and its two
# identity channels; the lattice rides along, so `instantiate(W)` needs no further arguments. Turning
# it into MPSKit's `FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian` lives in the MPSKit extension
# (ext/OpSumMPSKitExt.jl), on top of the Jordan emission of jordanmpo.jl.

using SparseArrays: SparseMatrixCSC
using TensorKit: Sector, dim

"""
    SymbolicMPO{I<:Sector, L<:AbstractLattice}

A compressed MPO whose entries are symbolic, as [`mpo`](@ref) returns it: `W[i]` is the sparse bond
matrix of site `i`, each stored entry a [`SiteOperator`](@ref) (a coefficient vector over the site's
[`Basis`](@ref)), and `W.bondsectors[i]` the charge of each index of the bond to the right of site `i`.
On a [`FiniteChain`](@ref) bond `0` is the trivial left boundary; on an [`InfiniteChain`](@ref) it is
bond `L`.

`W.start[i]` and `W.finish[i]` are the indices of the two identity channels on bond `i` ("nothing
placed yet" and "everything placed"), `0` where the compression spent no index on one; they are what
the Jordan form needs. Both are empty when they are unknown (decaying channels on a finite chain).

`instantiate(entry, basis)` gives one entry's tensor; `instantiate(W)` gives MPSKit's
`FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian` once MPSKit is loaded.
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

Compress `H` into a [`SymbolicMPO`](@ref) on its lattice: the same compression as
[`irrep_mpo`](@ref) (whose algorithm selectors `alg` it takes), with the identity channels identified
so that the result can be put in Jordan form. On a finite chain the finish channel is forced into the
cover, which can cost an index over `irrep_mpo` and is what makes the Jordan form at most `+2` per
bond over it.

```julia
W = mpo(opsum(FiniteChain(B, N), h))
instantiate(W[2][1, 3], B)          # one entry's tensor
using MPSKit; instantiate(W)        # an MPSKit.FiniteMPOHamiltonian
```
"""
function mpo(
        H::OperatorSum{I, <:FiniteChain},
        alg::Union{BipartiteAlgorithm, SVDBondAlgorithm} = BipartiteAlgorithm()
    ) where {I}
    lat = H.lattice
    isempty(H.channels) || return SymbolicMPO(lat, irrep_mpo(H, alg)..., Int[], Int[])
    tt = ITOTermTable(H.terms, length(lat))
    Ws, secs, starts, finishes = _irrep_channels(tt, nvertices(tt), bondstrategy(alg))
    return SymbolicMPO(lat, Ws, secs, starts, finishes)
end

function mpo(H::OperatorSum{I, <:InfiniteChain}, alg = BipartiteAlgorithm()) where {I}
    m = irrep_mpo(H, alg)
    return SymbolicMPO(H.lattice, m.Ws, m.bondsectors, m.start, m.done)
end

SymbolicMPO(lat::L, Ws, secs::Vector{Vector{I}}, start, finish) where {I, L} =
    SymbolicMPO{I, L}(lat, Ws, secs, start, finish)

"""
    mpo_terms(W::SymbolicMPO) -> Terms

The operator a finite [`SymbolicMPO`](@ref) represents, as a latticeless term bag (see
`mpo_terms(Ws, bondsectors)`).
"""
mpo_terms(W::SymbolicMPO{I, <:FiniteChain}) where {I} = mpo_terms(W.Ws, W.bondsectors)

"""
    irrep_mpo_tensors(W::SymbolicMPO) -> Vector{<:AbstractTensorMap}

The dense per-site tensors of `W`, each entry interpreted in its site's basis.
"""
irrep_mpo_tensors(W::SymbolicMPO{I, <:FiniteChain}) where {I} =
    irrep_mpo_tensors(W.Ws, W.bondsectors, W.lattice)
irrep_mpo_tensors(W::SymbolicMPO{I, <:InfiniteChain}) where {I} =
    irrep_mpo_tensors(InfiniteMPO(W.Ws, W.bondsectors, W.start, W.finish), W.lattice)

"""
    jordan_mpo_tensors(W::SymbolicMPO) -> Vector{<:SparseBlockTensorMap}

`W` in Jordan form, one `SparseBlockTensorMap` per site of the chain or of the unit cell: bond indices
ordered `(start, everything else, finish)`, identities at `(1, 1)` and `(end, end)` emitted as
`BraidingTensor`s, and each entry interpreted in its site's basis.
"""
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
    return jordan_mpo_tensors(W.Ws, W.bondsectors, W.start, W.finish, W.lattice)
end

# The unit cell: every bond carries both identity channels, so each is reordered as it stands, and
# the bond to the left of site 1 is bond `L`.
function jordan_mpo_tensors(W::SymbolicMPO{I, <:InfiniteChain}) where {I}
    L = length(W)
    bonds = [_jordan_bond(W.bondsectors[j], W.start[j], W.finish[j]) for j in 1:L]
    pos = Vector{Int}[first(bonds[mod1(j - 1, L)]) for j in 1:(L + 1)]
    jsec = Vector{I}[last(bonds[mod1(j - 1, L)]) for j in 1:(L + 1)]
    return _jordan_tensors(W.Ws, pos, jsec, W.lattice)
end

"""
    instantiate(W::SymbolicMPO)

MPSKit's `FiniteMPOHamiltonian` (on a [`FiniteChain`](@ref)) or `InfiniteMPOHamiltonian` (on an
[`InfiniteChain`](@ref)) for `W`, built from its Jordan form ([`jordan_mpo_tensors`](@ref)). Defined
by the MPSKit extension: load MPSKit first.
"""
instantiate(W::SymbolicMPO) = _mpskit_hamiltonian(W)

# Methods for `SymbolicMPO`s on either lattice come with the MPSKit extension.
_mpskit_hamiltonian(::Any) = throw(
    ArgumentError("instantiate(::SymbolicMPO) builds an MPSKit Hamiltonian: load MPSKit first (`using MPSKit`)")
)
