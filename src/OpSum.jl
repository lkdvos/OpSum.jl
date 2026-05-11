module OpSum

export Trie, Sum
export opsum, simplify
export mpo_bond_optimizations, BipartiteAlgorithm, SVDBondAlgorithm

using Dictionaries
using AbstractTrees
using SparseArraysBase: SparseArraysBase
using SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, storedpairs
using VectorInterface
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation
using LinearAlgebra: LinearAlgebra, kron
using LightSumTypes

# Utility
# -------
include("utility/linalg.jl")
include("utility/utility.jl")

# Data structures
# ---------------
include("datastructures/trie.jl")
include("datastructures/bipartite.jl")

# Operators
# ---------
include("operators/abstractoperators.jl")
include("operators/operatorbasis.jl")
include("operators/operatoralgebra.jl")
include("operators/globalalgebra.jl")

include("operators/paulioperators.jl")

# State machines
# --------------
include("statemachines/state_machines.jl")
include("statemachines/graphbuilding.jl")

end
