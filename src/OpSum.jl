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

# Data structures
# ---------------
include("sparsearrays.jl")
include("trie.jl")
include("dawg.jl")
include("dawgdict.jl")

include("linalg.jl")

# Operators
# ---------
include("operators.jl")
include("state_machines.jl")
include("compression.jl")

end
