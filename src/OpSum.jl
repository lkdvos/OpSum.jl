module OpSum

export Trie, SDAWG, DAWGDictionary
export @op_str, AbstractOperator, Operator, Scaled, Sum, Product
export opsum_vertex_operators, compress_vertex_operators

export ParallelDecomposition

using Dictionaries
using AbstractTrees
using SparseArraysBase: SparseArraysBase
using SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, storedpairs
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation
using QuantumOperatorAlgebra
using QuantumOperatorAlgebra: operatorstrings, begin_marker, end_marker, isbegin, isend
using LightSumTypes

# Utility
# -------
include("utility/linalg.jl")

# Data structures
# ---------------
include("datastructures/trie.jl")
include("datastructures/dawg.jl")
include("datastructures/dawgdict.jl")

# Operators
# ---------
# include("operators/localoperator.jl")
# using .LocalOperators
# import .LocalOperators: scalartype, flavourtype, isbegin, isend

# include("operators/symbolicoperator.jl")
# include("operators/operators.jl")

# State machines
# --------------
include("statemachines/state_machines.jl")
include("statemachines/compression.jl")

end
