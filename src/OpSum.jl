module OpSum

export Trie, SDAWG, DAWGDictionary
export @op_str, AbstractOperator, Operator, Scaled, Sum, Product
export opsum_vertex_operators, compress_vertex_operators, simplify, opsum
export TreeTopology, TTNOTerm, ttno_terms, ttno_bond_optimizations
export mpo_bond_optimizations

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

# Data structures
# ---------------
include("datastructures/trie.jl")
include("datastructures/dawg.jl")
include("datastructures/dawgdict.jl")
include("datastructures/graphnode.jl")
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
include("statemachines/compression.jl")
include("statemachines/ttno.jl")
include("statemachines/graphbuilding.jl")

end
