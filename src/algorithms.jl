# MPO bond-basis strategies and algorithm selectors
# =================================================
# Two layers, defined here (ahead of the pipeline) so both are nameable before the sweeps that
# consume them:
#
# * a **bond-basis strategy** — how one bond's compressed basis is chosen from that bond's (prefix,
#   suffix) coefficient data. This is the plug point of the reduced-MPO sweep (irrepgraph.jl): the
#   site step's other four phases are shared, only the basis choice differs.
# * an **algorithm selector** — the user-facing third argument of `irrep_mpo`, which names a
#   strategy (and, for the SVD, its truncation).

"""
    BondStrategy

Supertype of the reduced-MPO bond-basis strategies: [`VertexCover`](@ref),
[`IndependentSVD`](@ref) and [`SequentialSVD`](@ref). A strategy decides, at each bond, which
compressed basis the sweep keeps; everything else about the sweep is shared.
"""
abstract type BondStrategy end

"""
    VertexCover()

Bond-basis strategy: a minimum vertex cover of the bond's bipartite (prefix, suffix) graph, per
connected component. Lossless, and minimal among all MPOs with the same sparsity pattern. This is
what [`BipartiteAlgorithm`](@ref) selects.
"""
struct VertexCover <: BondStrategy end

"""
    IndependentSVD(trunc = nothing)

Bond-basis strategy: every bond is compressed **independently**, by an SVD of that bond's
coefficient matrix over the *raw* prefix/suffix classes of the term table.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` (e.g. `truncrank`, `trunctol`) or `nothing` for
the lossless default. Because the bonds do not see each other, `truncrank(k)` means **`k` retained
indices per bond**, whatever the neighbouring bonds did. Contrast [`SequentialSVD`](@ref).
"""
struct IndependentSVD{T} <: BondStrategy
    trunc::T
end
IndependentSVD() = IndependentSVD(nothing)

"""
    SequentialSVD(trunc = nothing)

Bond-basis strategy: a **sequential** left-to-right sweep (ITensor's QR-backend sweep), where each
bond is compressed in the basis left over from the bond before it.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` or `nothing` for the lossless default. Losslessly
this agrees with [`IndependentSVD`](@ref) — same per-sector bond dimensions, same operator — but
under truncation the two differ *by design*: here `truncrank(k)` means "`k` indices after whatever
upstream truncation already threw away", so an aggressive early truncation can starve the downstream
bonds of the states they would need. In exchange the sweep is incremental: it never re-derives a
bond's classes from scratch, and it reuses the persistent graph.
"""
struct SequentialSVD{T} <: BondStrategy
    trunc::T
end
SequentialSVD() = SequentialSVD(nothing)

"""
    BipartiteAlgorithm()

Algorithm selector: bipartite graph / minimum vertex cover (the default).

Lossless, and minimal among all MPOs with the same sparsity pattern. Selects the
[`VertexCover`](@ref) bond-basis strategy.
"""
struct BipartiteAlgorithm end

"""
    SVDBondAlgorithm(trunc = nothing; sweep = IndependentSVD)

Algorithm selector: SVD-based bond subspace selection.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` (e.g. `truncrank`, `trunctol`) or `nothing` for
the lossless default. Each bond's coefficient matrix is assembled as a charge-graded `TensorMap`, so
`svd_trunc` does the per-sector SVD *and* the truncation globally across sectors, respecting the
quantum dimensions.

`sweep` picks the truncation semantics, and only matters when `trunc !== nothing`:

| `sweep` | meaning of `truncrank(k)` |
|---|---|
| [`IndependentSVD`](@ref) (default) | `k` retained indices **per bond**; every bond is compressed on the raw prefix/suffix classes, independently of its neighbours |
| [`SequentialSVD`](@ref) | `k` retained indices **after upstream truncation**; each bond is compressed in the basis left over from the previous one, so an aggressive early truncation can starve downstream bonds |

Both are exact when `trunc === nothing`, and then produce the same per-sector bond dimensions.
`SVDBondAlgorithm(alg::BondStrategy)` names the strategy (and its truncation) directly.
"""
struct SVDBondAlgorithm{S <: BondStrategy}
    strategy::S
end
SVDBondAlgorithm(trunc; sweep = IndependentSVD) = SVDBondAlgorithm(sweep(trunc))
SVDBondAlgorithm(; sweep = IndependentSVD) = SVDBondAlgorithm(sweep(nothing))

# The strategy an algorithm selector names. `irrep_mpo` dispatches the sweep on this.
bondstrategy(::BipartiteAlgorithm) = VertexCover()
bondstrategy(alg::SVDBondAlgorithm) = alg.strategy
