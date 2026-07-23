# MPO bond-optimization algorithm selectors
# ==========================================
# Shared by both the dense (`mpo_bond_optimizations`, graphbuilding.jl) and the symmetric/irrep
# (`irrep_mpo`, irrepmpo.jl) pipelines, so they are defined here — ahead of both — rather than in
# either pipeline's file.

"""Algorithm selector: bipartite graph / minimum vertex cover (current default)."""
struct BipartiteAlgorithm end

"""Algorithm selector: SVD-based bond subspace selection.

`trunc` is a `MatrixAlgebraKit.TruncationStrategy` (e.g. `truncrank`, `trunctol`) or `nothing` for
the lossless default. The same strategy drives the dense path (`svd_trunc!` on a plain coefficient
matrix) and the irrep path (`svd_trunc` on a charge-graded coefficient `TensorMap`, where truncation
is applied globally across charge sectors)."""
struct SVDBondAlgorithm
    trunc  # TruncationStrategy or nothing
end
SVDBondAlgorithm() = SVDBondAlgorithm(nothing)
