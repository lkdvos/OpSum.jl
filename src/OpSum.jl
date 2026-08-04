module OpSum

# Public API. Kept here rather than next to each definition so that `using OpSum` has one, greppable
# source of truth — the surface was previously spread over four files, which is why every example and
# test opened with a long `using OpSum: …` list for things that were already exported (or were not,
# with no way to tell which).

# on-site operators (memoised, so they may be built inside a term loop)
export IrrepOperator, spin, scalarop, project, matrixunit, spin_ops, fermion_ops
# term algebra: place, couple, combine, bind to a lattice
export TermSum, TermList, couple, hc, onlattice, lattice
# MPO construction
export irrep_mpo, irrep_mpo_tensors, jordan_mpo_tensors, mpo_terms, instantiate
export BipartiteAlgorithm, SVDBondAlgorithm
export BondStrategy, VertexCover, IndependentSVD, SequentialSVD
# verification
export islossless, mpo_tensormap

using Dictionaries
using SparseArrays: SparseMatrixCSC, sparse, nonzeros, nzrange, rowvals
using VectorInterface
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation
using LinearAlgebra: LinearAlgebra

# Algorithm selectors (shared by the dense and irrep pipelines)
# -------------------------------------------------------------
include("algorithms.jl")

# Utility
# -------
include("utility/linalg.jl")
include("utility/memo.jl")

# Data structures
# ---------------
include("datastructures/bipartite.jl")
include("datastructures/connectedcomponents.jl")

# Operators
# ---------
include("operators/operatorbasis.jl")

include("operators/irreptensoroperators.jl")
using .IrrepTensorOperators: IrrepOperator
include("operators/irrepkey.jl")
include("operators/onsiteop.jl")
include("operators/irrepalgebra.jl")
include("operators/irrepprojection.jl")
include("operators/builders.jl")
include("operators/irreptermtable.jl")
include("operators/irrepgraph.jl")
include("operators/irrepmpo.jl")
include("operators/jordanmpo.jl")

end
