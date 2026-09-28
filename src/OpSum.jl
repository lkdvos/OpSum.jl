module OpSum

# Public API
# ----------

# on-site operators: build them once, outside any loop
export IrrepOperator, spin, scalarop, project, matrixunit, spin_ops, fermion_ops
# term algebra: place and couple. Latticeless — the lattice is supplied at `irrep_mpo`
export Term, Terms, couple, couple_channels, opsum, canonicalize!
# lattices: the two things an operator can be compressed over
export AbstractLattice, FiniteChain, InfiniteChain
# MPO construction
export irrep_mpo, irrep_mpo_tensors, jordan_mpo_tensors, mpo_terms, instantiate
export FiniteMPO, InfiniteMPO
export BipartiteAlgorithm, SVDBondAlgorithm
export BondStrategy, VertexCover, IndependentSVD, SequentialSVD
# exponentially decaying interactions
export expterm, ExpSum, MixedSum
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
include("operators/compression/algorithms.jl")

# Utility
# -------
include("utility/linalg.jl")
include("utility/memo.jl")

# Data structures
# ---------------
include("datastructures/bipartite.jl")
include("datastructures/connectedcomponents.jl")

# Lattices
# --------
include("operators/lattices.jl")

# Operators — symbolic algebra
# -----------------------------
include("operators/algebra/operatorbasis.jl")

include("operators/algebra/irreptensoroperators.jl")
using .IrrepTensorOperators: IrrepOperator
include("operators/algebra/irrepkey.jl")
include("operators/algebra/siteoperator.jl")
include("operators/algebra/irrepalgebra.jl")
include("operators/algebra/irrepinstantiate.jl")
include("operators/algebra/irrepprojection.jl")
include("operators/algebra/builders.jl")

# Operators — compression to a reduced MPO
# ------------------------------------------
include("operators/compression/irreptermtable.jl")
include("operators/infinite/expterms.jl")
include("operators/compression/irrepinterning.jl")
include("operators/compression/irrepgraph.jl")
include("operators/compression/irrepgraph_vc.jl")
include("operators/compression/irrepgraph_svd.jl")
include("operators/infinite/infinitechain.jl")
include("operators/infinite/infinitegraph.jl")
include("operators/compression/irrepmpo.jl")
include("operators/compression/jordanmpo.jl")

end
