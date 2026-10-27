module OpSum

# on-site operators
export Basis
export spin_basis, fermion_basis, boson_basis, hubbard_basis, tj_basis
export boson_ops, hubbard_ops, tj_ops
export Letter, scalarop, project, spin_ops, fermion_ops
# unplaced K-site operators
export LocalOperator
# term algebra (latticeless); `Terms(args...)` collects terms
export Term, Terms, couple, couple_channels
# the lattice-carrying container; `opsum(lat, args...)` builds one
export OperatorSum, opsum, opsum!
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
    instantiate(H::OperatorSum)
    instantiate(W::SymbolicMPO)
    instantiate(W::TaylorMPO, τ::Number)

Materialize a symbolic object. A letter or on-site operator in a [`Basis`](@ref OpSum.Basis), or an
`OperatorSum` on its lattice, becomes a `TensorMap`. With MPSKit loaded (`using MPSKit`), a
[`SymbolicMPO`](@ref) becomes MPSKit's `FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian`, built from its
Jordan form ([`jordan_mpo_tensors`](@ref)), and a [`TaylorMPO`](@ref) its `FiniteMPO` or `InfiniteMPO` of
`exp(τH)` evaluated at `τ` (see [`mpo_tensors`](@ref)); `make_time_mpo(H::OperatorSum, dt,
TaylorCluster(; N, extension, compression))` builds `taylor_mpo(H, N; extension, compression)` and
instantiates it at `τ = -dt` (imaginary evolution) or `τ = -i dt`.
"""
function instantiate end

include("algebra/letters.jl")
include("algebra/siteoperator.jl")
include("algebra/terms.jl")
include("algebra/recouple.jl")
include("algebra/couple.jl")
include("algebra/instantiate.jl")
include("algebra/localoperator.jl")
include("algebra/basis.jl")
include("algebra/projection.jl")
include("lattices.jl")
include("algebra/adjoint.jl")
include("algebra/products.jl")
include("algebra/letterproducts.jl")
include("algebra/stockbases.jl")

include("compression/termtable.jl")
include("expterms.jl")
include("compression/channels.jl")
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

function __init__()
    # `instantiate` of the symbolic MPOs is a method of the MPSKit extension
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === instantiate && !isempty(argtypes) && argtypes[1] <: Union{SymbolicMPO, TaylorMPO} &&
                Base.get_extension(OpSum, :OpSumMPSKitExt) === nothing
            print(io, "\ninstantiate of a ", nameof(argtypes[1]), " builds MPSKit MPOs: load MPSKit first (`using MPSKit`).")
        end
        return nothing
    end
    return nothing
end

end
