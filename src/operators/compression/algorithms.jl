# Bond-basis strategies of the sweep and the `mpo` algorithm selectors that name them.

"""
    BondStrategy

Supertype of [`VertexCover`](@ref), [`IndependentSVD`](@ref) and [`SequentialSVD`](@ref): how the
sweep chooses each bond's compressed basis.
"""
abstract type BondStrategy end

"""
    VertexCover()

Minimum vertex cover of each bond's bipartite (prefix, suffix) graph, per connected component.
Lossless; what [`BipartiteAlgorithm`](@ref) selects.
"""
struct VertexCover <: BondStrategy end

"""
    IndependentSVD(trunc = nothing)

Every bond is compressed independently by an SVD over the raw prefix/suffix classes. `trunc` is a
`MatrixAlgebraKit.TruncationStrategy` or `nothing` (lossless); `truncrank(k)` keeps `k` indices per bond.
"""
struct IndependentSVD{T} <: BondStrategy
    trunc::T
end
IndependentSVD() = IndependentSVD(nothing)

"""
    SequentialSVD(trunc = nothing)

Left-to-right sweep compressing each bond in the basis left by the previous one, so `truncrank(k)` keeps
`k` indices after upstream truncation. Agrees with [`IndependentSVD`](@ref) when lossless.
"""
struct SequentialSVD{T} <: BondStrategy
    trunc::T
end
SequentialSVD() = SequentialSVD(nothing)

"""
    BipartiteAlgorithm()

Algorithm selector for [`VertexCover`](@ref) (the default).
"""
struct BipartiteAlgorithm end

"""
    SVDBondAlgorithm(trunc = nothing; sweep = IndependentSVD)

Algorithm selector for SVD bond truncation (`trunc` as in [`IndependentSVD`](@ref); `svd_trunc`
truncates globally across charge sectors). `sweep` is [`IndependentSVD`](@ref) or
[`SequentialSVD`](@ref), which differ in the meaning of `truncrank(k)`; `SVDBondAlgorithm(strategy)`
names a strategy directly.
"""
struct SVDBondAlgorithm{S <: BondStrategy}
    strategy::S
end
SVDBondAlgorithm(trunc; sweep = IndependentSVD) = SVDBondAlgorithm(sweep(trunc))
SVDBondAlgorithm(; sweep = IndependentSVD) = SVDBondAlgorithm(sweep(nothing))

bondstrategy(::BipartiteAlgorithm) = VertexCover()
bondstrategy(alg::SVDBondAlgorithm) = alg.strategy
