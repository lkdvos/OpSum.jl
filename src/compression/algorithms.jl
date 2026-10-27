# Bond-basis strategies of the sweep: the algorithm argument of `mpo`.

"""
    BondStrategy

Supertype of [`VertexCover`](@ref), [`IndependentSVD`](@ref) and [`SequentialSVD`](@ref): how the
sweep chooses each bond's compressed basis. It is the optional second argument of [`mpo`](@ref).
"""
abstract type BondStrategy end

"""
    VertexCover()

Minimum vertex cover of each bond's bipartite (prefix, suffix) graph, per connected component. Lossless;
the default of [`mpo`](@ref), and the only strategy available on an `InfiniteChain` or with decaying
channels.
"""
struct VertexCover <: BondStrategy end

"""
    IndependentSVD(trunc = notrunc())

Every bond is compressed independently by an SVD over the raw prefix/suffix classes. `trunc` is a
`MatrixAlgebraKit.TruncationStrategy`; the default `notrunc()` is lossless, `truncrank(k)` keeps `k`
indices per bond, and `svd_trunc`-style strategies truncate globally across charge sectors. The default
`notrunc()` still truncates at `eps(Float64)` relative to the norm of the singular values, to discard round-off.
Finite chains without decaying channels only.
"""
struct IndependentSVD{T <: TruncationStrategy} <: BondStrategy
    trunc::T
end
IndependentSVD() = IndependentSVD(notrunc())

"""
    SequentialSVD(trunc = notrunc())

Left-to-right sweep compressing each bond in the basis left by the previous one, so `truncrank(k)` keeps
`k` indices after upstream truncation (`trunc` as in [`IndependentSVD`](@ref)). When lossless it has the
same bond dimensions as [`IndependentSVD`](@ref), but not the same gauge. Finite chains without decaying
channels only.
"""
struct SequentialSVD{T <: TruncationStrategy} <: BondStrategy
    trunc::T
end
SequentialSVD() = SequentialSVD(notrunc())

# does the strategy truncate (and so need the basis' own metric, metric.jl)?
truncates(::VertexCover) = false
truncates(s::Union{IndependentSVD, SequentialSVD}) = !(s.trunc isa NoTruncation)
