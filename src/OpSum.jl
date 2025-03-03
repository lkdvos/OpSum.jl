module OpSum

export Trie
export SDAWG
export DAWGDictionary
export opsum_to_fsm

export ParallelDecomposition

using Dictionaries
using SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, storedpairs
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation

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
include("operators/operators.jl")

# State machines
# --------------
include("statemachines/state_machines.jl")
include("statemachines/compression.jl")

end
