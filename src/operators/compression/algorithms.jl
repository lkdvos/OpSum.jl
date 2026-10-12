# MPO bond-basis strategies (how one bond's compressed basis is chosen; the plug point of the sweep in
# irrepgraph.jl) and the user-facing algorithm selectors of `mpo` that name them.

"""
    BondStrategy

Supertype of the reduced-MPO bond-basis strategies [`VertexCover`](@ref), [`IndependentSVD`](@ref) and
[`SequentialSVD`](@ref): each decides which compressed basis the sweep keeps at a bond.
"""
abstract type BondStrategy end

"""
    VertexCover()

Bond-basis strategy: a minimum vertex cover of the bond's bipartite (prefix, suffix) graph, per
connected component. Lossless and minimal among MPOs with the same sparsity pattern; what
[`BipartiteAlgorithm`](@ref) selects.
"""
struct VertexCover <: BondStrategy end

"""
    IndependentSVD(trunc = nothing)

Bond-basis strategy: every bond is compressed **independently**, by an SVD of its coefficient matrix over
the *raw* prefix/suffix classes of the term table.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` (e.g. `truncrank`, `trunctol`) or `nothing` (lossless).
`truncrank(k)` means **`k` retained indices per bond**. Contrast [`SequentialSVD`](@ref).
"""
struct IndependentSVD{T} <: BondStrategy
    trunc::T
end
IndependentSVD() = IndependentSVD(nothing)

"""
    SequentialSVD(trunc = nothing)

Bond-basis strategy: a **sequential** left-to-right sweep, where each bond is compressed in the basis left
over from the bond before it.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` or `nothing` (lossless). Losslessly this agrees with
[`IndependentSVD`](@ref); under truncation the two differ by design: here `truncrank(k)` means "`k`
indices after whatever upstream truncation already threw away", so an aggressive early truncation can
starve downstream bonds. In exchange the sweep is incremental and reuses the persistent graph.
"""
struct SequentialSVD{T} <: BondStrategy
    trunc::T
end
SequentialSVD() = SequentialSVD(nothing)

"""
    BipartiteAlgorithm()

Algorithm selector: bipartite graph / minimum vertex cover (the default). Selects [`VertexCover`](@ref).
"""
struct BipartiteAlgorithm end

"""
    SVDBondAlgorithm(trunc = nothing; sweep = IndependentSVD)

Algorithm selector: SVD-based bond subspace selection.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` (e.g. `truncrank`, `trunctol`) or `nothing` (lossless).
Each bond's coefficient matrix is a charge-graded `TensorMap`, so `svd_trunc` truncates globally across
sectors, respecting the quantum dimensions.

`sweep` picks the truncation semantics (it only matters when `trunc !== nothing`):

| `sweep` | meaning of `truncrank(k)` |
|---|---|
| [`IndependentSVD`](@ref) (default) | `k` retained indices **per bond**; every bond is compressed on the raw prefix/suffix classes, independently of its neighbours |
| [`SequentialSVD`](@ref) | `k` retained indices **after upstream truncation**; each bond is compressed in the basis left over from the previous one, so an aggressive early truncation can starve downstream bonds |

Both are exact when `trunc === nothing`. `SVDBondAlgorithm(alg::BondStrategy)` names the strategy directly.
"""
struct SVDBondAlgorithm{S <: BondStrategy}
    strategy::S
end
SVDBondAlgorithm(trunc; sweep = IndependentSVD) = SVDBondAlgorithm(sweep(trunc))
SVDBondAlgorithm(; sweep = IndependentSVD) = SVDBondAlgorithm(sweep(nothing))

bondstrategy(::BipartiteAlgorithm) = VertexCover()
bondstrategy(alg::SVDBondAlgorithm) = alg.strategy
