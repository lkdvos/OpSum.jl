module OpSum

export Sum
export opsum, simplify
export mpo_bond_optimizations, BipartiteAlgorithm, SVDBondAlgorithm

using Dictionaries
using SparseArraysBase: SparseArraysBase
using SparseArraysBase: SparseArrayDOK, SparseMatrixDOK, storedpairs
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

# Operators
# ---------
include("operators/abstractoperators.jl")
include("operators/operatorbasis.jl")
include("operators/operatoralgebra.jl")
include("operators/globalalgebra.jl")
include("operators/termtable.jl")

include("operators/paulioperators.jl")
include("operators/irreptensoroperators.jl")
using .IrrepTensorOperators: IrrepOperator
include("operators/irrepalgebra.jl")
include("operators/irrepkey.jl")
include("operators/irreptermtable.jl")
include("operators/irrepmpo.jl")

# State machines
# --------------
include("statemachines/state_machines.jl")
include("statemachines/graphbuilding.jl")

end
