module OpSum

# Public API
# ----------

# on-site operators: build them once, outside any loop
export IrrepOperator, spin, scalarop, project, matrixunit
# term algebra: place, couple, combine
export TermSum, couple
# MPO construction
export irrep_mpo, irrep_mpo_tensors, mpo_terms, instantiate
export BipartiteAlgorithm, SVDBondAlgorithm
# symbolic-algebra scaffolding (slated for removal — see research/onsite-products.md)
export Sum, simplify

using Dictionaries
using SparseArrays: SparseMatrixCSC, sparse, nonzeros, nzrange, rowvals
using VectorInterface
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation
using LinearAlgebra: LinearAlgebra, kron
using LightSumTypes

# Algorithm selectors (shared by the dense and irrep pipelines)
# -------------------------------------------------------------
include("algorithms.jl")

# Utility
# -------
include("utility/linalg.jl")
include("utility/utility.jl")

# Data structures
# ---------------
include("datastructures/bipartite.jl")
include("datastructures/connectedcomponents.jl")

# Operators
# ---------
include("operators/abstractoperators.jl")
include("operators/operatorbasis.jl")
include("operators/operatoralgebra.jl")

include("operators/irreptensoroperators.jl")
using .IrrepTensorOperators: IrrepOperator
include("operators/irrepalgebra.jl")
include("operators/irrepkey.jl")
include("operators/irrepprojection.jl")
include("operators/irreptermtable.jl")
include("operators/irrepgraph.jl")
include("operators/irrepmpo.jl")

end
