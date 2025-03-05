module OpSum

export Trie, SDAWG, DAWGDictionary
export @op_str, AbstractOperator, Operator, Scaled, Sum, Product

export ParallelDecomposition

using Dictionaries
using SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, storedpairs
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation
using Moshi.Data: @data
using Moshi.Match: @match

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
include("operators/symbolicoperator.jl")
include("operators/operators.jl")

# State machines
# --------------
include("statemachines/state_machines.jl")
include("statemachines/compression.jl")

end
