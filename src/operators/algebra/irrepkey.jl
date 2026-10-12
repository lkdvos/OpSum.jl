# `ITOKey = (op, bond_charge_out, vertex)`. `Term` stores keys, not a fusion tree: for a caterpillar the
# keys *are* the tree (`bondcharges` projects onto them, `_tree_from_bonds` inverts it). The total alone
# would not do: for K ≥ 3 non-abelian several channels reach the same total.

using TensorKit
using TensorKit: Sector, FusionTree, fusiontrees, unit
using .IrrepTensorOperators: IrrepOperator

# the identity letter, element 1 of every `Basis`; fills idle sites, and placement drops it
passthrough(::Type{I}) where {I <: Sector} = IrrepOperator{I}(unit(I), 1)

ispassthrough(op::IrrepOperator) = op.n == 1 && op.c == unit(op.c)

# `bond` is the running fusion charge out of the site, `vertex` the multiplicity label (1: multiplicity-free)
struct ITOKey{I <: Sector}
    op::IrrepOperator{I}
    bond::I
    vertex::Int
end

function Base.isless(x::ITOKey{I}, y::ITOKey{I}) where {I}
    x.op == y.op || return isless(x.op, y.op)
    x.bond == y.bond || return isless(x.bond, y.bond)
    return isless(x.vertex, y.vertex)
end
Base.:(==)(x::ITOKey{I}, y::ITOKey{I}) where {I} =
    x.op == y.op && x.bond == y.bond && x.vertex == y.vertex
Base.hash(x::ITOKey, h::UInt) = hash(x.vertex, hash(x.bond, hash(x.op, hash(:ITOKey, h))))

function Base.show(io::IO, k::ITOKey)
    return print(io, "ITOKey(", k.op, ", bond=", k.bond, ", μ=", k.vertex, ")")
end

# all left-nested fusion trees of `charges` fusing to `total`, in TensorKit's canonical order
function caterpillar_trees(charges::NTuple{K, I}, total::I) where {K, I <: Sector}
    K == 0 && throw(ArgumentError("empty charge tuple"))
    return collect(fusiontrees(charges, total, ntuple(_ -> false, K)))
end

# running bond charges `b₁…b_K`: `uncoupled[1]`, the inner lines, `coupled`
function bondcharges(f::FusionTree{I, K}) where {I, K}
    K == 0 && return I[]
    K == 1 && return I[f.coupled]
    return I[f.uncoupled[1], f.innerlines..., f.coupled]
end
