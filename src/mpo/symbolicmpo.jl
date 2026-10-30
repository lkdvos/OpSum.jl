# `SymbolicMPO` and `mpo`: the finite compression behind it, `mpo_terms` (its inverse on the reduced level) and
# the verification `islossless`.

using SparseArrays: SparseMatrixCSC
using TensorKit: Sector, dim, unit

"""
    SymbolicMPO{I<:Sector, L<:AbstractLattice}

A compressed MPO with symbolic entries, as [`mpo`](@ref) returns it: `W[i]` is the sparse bond matrix of
site `i` (entries are [`SiteOperator`](@ref)s) and `W.bondsectors[i]` the charge of each index of the bond
to its right (on an [`InfiniteChain`](@ref) bond `L` is the one left of site 1). `W.start[i]` and
`W.finish[i]` are the two identity-channel indices on bond `i` (`0` if the compression spent none).

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
finish channel is forced into the cover for that.

`alg` is a [`BondStrategy`](@ref): [`VertexCover`](@ref) (default, lossless minimum vertex cover),
[`IndependentSVD`](@ref) or [`SequentialSVD`](@ref) (finite chains without channels only).
"""
function mpo(H::OperatorSum{I, L}, alg::BondStrategy = VertexCover()) where {I, L <: FiniteChain}
    N = length(H.lattice)
    if !isempty(H.channels)
        alg isa VertexCover || throw(
            ArgumentError(
                "SVD strategies cannot compress an exponentially decaying interaction: they need the " *
                    "whole bond coefficient matrix up front. Use VertexCover() (the default)."
            )
        )
        compressed = sweep(TermTable(H.terms), N, alg; channels = lower_channels(H.channels, 1))
    elseif truncates(alg)
        compressed = compress_in_metric(H.terms, H.lattice, alg)   # a truncation has to see the basis' own inner product (metric.jl)
    else
        compressed = sweep(TermTable(H.terms), N, alg)
    end
    (; Ws, bondsectors, start, finish) = compressed
    return SymbolicMPO{I, L}(H.lattice, Ws, bondsectors, start, finish)
end

mpo(H::OperatorSum{I, <:InfiniteChain}, ::VertexCover = VertexCover()) where {I} =
    infinite_window(unitcell_terms(H))

mpo(::OperatorSum{I, <:InfiniteChain}, ::Union{IndependentSVD, SequentialSVD}) where {I} = throw(
    ArgumentError(
        "SVD strategies do not extend to infinite chains: they compress each bond " *
            "independently against a vacuum-terminated layout, with no bond basis that closes " *
            "on itself. Use VertexCover() (the default)."
    )
)

mpo(::Union{Term, Terms, ExpSum}, args...) = throw(
    ArgumentError("mpo takes an OperatorSum: put the terms on a lattice first, `mpo(opsum(lat, h))`")
)

"""
    mpo_terms(W::SymbolicMPO) -> Terms

Reconstruct the operator `W` generates, as a latticeless [`Terms`](@ref): enumerate every path through the
bond matrices and read off the letters with their outgoing bond charges. `mpo_terms(mpo(H)) ≈ H.terms` is
[`islossless`](@ref). Finite chains only.
"""
mpo_terms(W::SymbolicMPO{I, <:FiniteChain}) where {I} = reduced_terms(W.Ws, W.bondsectors)

# The paths of the bond matrices `Ws` from index `leftidx` of the first bond to index `rightidx` of the last
# (default `1` and "any"); for an infinite MPO tiled over a window they are the two identity channels.
function reduced_terms(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}};
        leftidx::Int = 1, rightidx::Union{Nothing, Int} = nothing
    ) where {I}
    N = length(Ws)
    N == 0 && return Terms{I}()
    cols = Term{I}[]
    function walk(i, l0, coeff, sites, keys)
        if i > N
            (rightidx === nothing || l0 == rightidx) &&
                push!(cols, Term{I}(copy(sites), copy(keys), coeff))
            return
        end
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            l == l0 || continue
            for (letter, c) in pairs(localop)
                pass = ispassthrough(letter)
                pass || (push!(sites, i); push!(keys, Factor{I}(letter, bondsectors[i][r])))
                walk(i + 1, r, coeff * c, sites, keys)
                pass || (pop!(sites); pop!(keys))
            end
        end
        return
    end
    walk(1, leftidx, ComplexF64(1), Int[], Factor{I}[])
    return Terms{I}(cols)
end

"""
    islossless(H::OperatorSum[, alg]) -> Bool

Whether the compressed MPO reproduces `H` exactly: `mpo_terms(mpo(H)) ≈ chain_terms(H)`. Purely symbolic,
finite chains only; after a truncating SVD [`BondStrategy`](@ref) `false` is expected.
"""
function islossless(H::OperatorSum{I, <:FiniteChain}, args...) where {I}
    return mpo_terms(mpo(H, args...)) ≈ chain_terms(H)
end

islossless(::OperatorSum{I, <:InfiniteChain}, args...) where {I} = throw(
    ArgumentError("islossless is not supported on an infinite chain")
)
