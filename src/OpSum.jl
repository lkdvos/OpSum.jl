module OpSum

# on-site operators
export Basis
export spin_basis, fermion_basis, boson_basis, hubbard_basis, tj_basis
export boson_ops, hubbard_ops, tj_ops
export IrrepOperator, scalarop, project, spin_ops, fermion_ops
# unplaced K-site operators
export LocalOperator
# term algebra (latticeless)
export Term, Terms, couple, couple_channels, opsum, canonicalize!
# the lattice-carrying container
export OperatorSum, opsum!
# lattices
export AbstractLattice, FiniteChain, InfiniteChain
# MPO construction
export mpo, SymbolicMPO
export irrep_mpo_tensors, jordan_mpo_tensors, mpo_terms, instantiate
export BipartiteAlgorithm, SVDBondAlgorithm
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
using MatrixAlgebraKit: AbstractAlgorithm, TruncationStrategy, NoTruncation
using LinearAlgebra: LinearAlgebra

include("operators/compression/algorithms.jl")

include("utility/linalg.jl")
include("utility/memo.jl")

include("datastructures/bipartite.jl")
include("datastructures/connectedcomponents.jl")

include("operators/lattices.jl")

"""
    instantiate(op, V)

Materialize a symbolic operator as a `TensorMap` on the physical space `V`.
"""
function instantiate end

include("operators/algebra/irreptensoroperators.jl")
using .IrrepTensorOperators: IrrepOperator
include("operators/algebra/irrepkey.jl")
include("operators/algebra/siteoperator.jl")
include("operators/algebra/irrepalgebra.jl")
include("operators/algebra/irrepinstantiate.jl")
include("operators/algebra/localoperator.jl")
include("operators/algebra/irrepprojection.jl")
include("operators/algebra/basis.jl")
include("operators/algebra/builders.jl")
include("operators/algebra/products.jl")
include("operators/algebra/letterproducts.jl")
include("operators/algebra/stockbases.jl")

include("operators/compression/irreptermtable.jl")
include("operators/infinite/expterms.jl")
include("operators/operatorsum.jl")
include("operators/compression/symbolicmpo.jl")
include("operators/compression/irrepinterning.jl")
include("operators/compression/irrepgraph.jl")
include("operators/compression/irrepgraph_vc.jl")
include("operators/compression/irrepgraph_svd.jl")
include("operators/compression/metric.jl")
include("operators/infinite/infinitechain.jl")
include("operators/infinite/infinitegraph.jl")
include("operators/compression/irrepmpo.jl")
include("operators/compression/jordanmpo.jl")
include("operators/timeevolution/jordanlayout.jl")
include("operators/timeevolution/kronpower.jl")
include("operators/timeevolution/taylor.jl")
include("operators/timeevolution/taylormpo.jl")

end
