module OpSum

# on-site operators
export Basis
export spin_basis, fermion_basis, boson_basis, hubbard_basis, tj_basis
export boson_ops, hubbard_ops, tj_ops
export Letter, scalarop, project, spin_ops, fermion_ops
# unplaced K-site operators
export LocalOperator
# term algebra (latticeless)
export Term, Terms, couple, couple_channels, opsum
# the lattice-carrying container
export OperatorSum, opsum!
# lattices
export AbstractLattice, FiniteChain, InfiniteChain
# MPO construction
export mpo, SymbolicMPO
export mpo_tensors, jordan_mpo_tensors, mpo_terms, instantiate
export BondStrategy, VertexCover, IndependentSVD, SequentialSVD
# time evolution
export taylor_mpo, TaylorMPO
# exponentially decaying interactions
export expterm, ExpSum
# verification
export islossless, mpo_tensormap

using Dictionaries
using SparseArrays: SparseMatrixCSC, sparse, nonzeros, nzrange, rowvals
using VectorInterface
using MatrixAlgebraKit
using MatrixAlgebraKit: TruncationStrategy, NoTruncation
using LinearAlgebra: LinearAlgebra

include("utility/helpers.jl")
include("utility/bipartite.jl")

"""
    instantiate(op, B)

Materialize a symbolic operator as a `TensorMap`: a letter or on-site operator in a [`Basis`](@ref OpSum.Basis),
or an `OperatorSum` on its lattice.
"""
function instantiate end

include("algebra/letters.jl")
include("algebra/siteoperator.jl")
include("algebra/terms.jl")
include("algebra/couple.jl")
include("algebra/instantiate.jl")
include("algebra/localoperator.jl")
include("algebra/projection.jl")
include("algebra/basis.jl")
include("lattices.jl")
include("algebra/adjoint.jl")
include("algebra/products.jl")
include("algebra/letterproducts.jl")
include("algebra/stockbases.jl")

include("compression/termtable.jl")
include("expterms.jl")
include("operatorsum.jl")
include("compression/algorithms.jl")
include("mpo/symbolicmpo.jl")
include("compression/graph.jl")
include("compression/graph_vc.jl")
include("compression/graph_svd.jl")
include("compression/metric.jl")
include("infinite/unitcell.jl")
include("infinite/window.jl")
include("mpo/tensors.jl")
include("mpo/jordan.jl")
include("timeevolution/kronpower.jl")
include("timeevolution/taylormpo.jl")
include("timeevolution/taylor.jl")

end
