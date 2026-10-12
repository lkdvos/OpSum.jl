# Charge-augmented alphabet + caterpillar fusion helpers: `ITOKey = (op, bond_charge_out, vertex)`.
# Prefixes agreeing on the full `ITOKey` share automaton state, so the automaton is block-diagonal in
# the bond charge.
#
# `Term` stores keys, not a fusion tree: for a caterpillar the keys *are* the tree (`bondcharges`
# projects one onto them; `_tree_from_bonds` inverts it). `total` alone would not do,
# since for `K ≥ 3` non-abelian several channels reach the same total.

using TensorKit
using TensorKit: Sector, FusionTree, fusiontrees, unit
using .IrrepTensorOperators: IrrepOperator

"""
    passthrough(::Type{I}) where {I<:Sector}

The pass-through symbol: the identity, element 1 of every [`Basis`](@ref) (trivial charge `unit(I)`,
index `1`). Fills idle sites. Placement drops it, so a placed `Term` never contains it.
"""
passthrough(::Type{I}) where {I <: Sector} = IrrepOperator{I}(unit(I), 1)

"""
    ispassthrough(op::IrrepOperator)

Whether `op` is the identity, i.e. the pass-through letter `(unit, 1)`.
"""
ispassthrough(op::IrrepOperator) = op.n == 1 && op.c == unit(op.c)

"""
    ITOKey{I<:Sector}

Alphabet symbol of the charge-augmented automaton: `op` (the ITO letter or pass-through), `bond`
(the running fusion charge *out* of this site) and `vertex` (the fusion-vertex multiplicity label,
`1` for multiplicity-free fusion).
"""
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

"""
    caterpillar_trees(charges::NTuple{K,I}, total::I) -> Vector{FusionTree{I,K}}

All left-nested caterpillar fusion trees of the per-site `charges` fusing to `total`, in
TensorKit's canonical order.
"""
function caterpillar_trees(charges::NTuple{K, I}, total::I) where {K, I <: Sector}
    K == 0 && throw(ArgumentError("empty charge tuple"))
    return collect(fusiontrees(charges, total, ntuple(_ -> false, K)))
end

"""
    bondcharges(f::FusionTree{I,K})

Running bond charges `b₁…b_K` of the caterpillar tree (`b₁ = uncoupled[1]`, `bₖ = innerlines[k-1]`
for `2 ≤ k ≤ K-1`, `b_K = coupled`); element `k` is the bond charge *out* of active position `k`.
"""
function bondcharges(f::FusionTree{I, K}) where {I, K}
    K == 0 && return I[]
    K == 1 && return I[f.coupled]
    return I[f.uncoupled[1], f.innerlines..., f.coupled]
end
