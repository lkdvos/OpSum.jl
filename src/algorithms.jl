# MPO bond-optimization algorithm selectors
# ==========================================
# How `irrep_mpo` (irrepmpo.jl) chooses each bond's compressed basis. Defined here, ahead of the
# pipeline, so the selectors are nameable before the sweeps that consume them.

"""Algorithm selector: bipartite graph / minimum vertex cover (the default).

Lossless, and minimal among all MPOs with the same sparsity pattern. Routes to the persistent-graph
sweep `_irrep_graph_bipartite` (irrepgraph.jl)."""
struct BipartiteAlgorithm end

"""Algorithm selector: SVD-based bond subspace selection.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` (e.g. `truncrank`, `trunctol`) or `nothing` for
the lossless default. Each bond's coefficient matrix is assembled as a charge-graded `TensorMap`, so
`svd_trunc` does the per-sector SVD *and* the truncation globally across sectors, respecting the
quantum dimensions."""
struct SVDBondAlgorithm
    trunc  # TruncationStrategy or nothing
end
SVDBondAlgorithm() = SVDBondAlgorithm(nothing)
