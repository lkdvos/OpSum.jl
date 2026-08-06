# Charge-augmented alphabet + caterpillar fusion helpers
# ======================================================
# The alphabet symbol `ITOKey = (op, bond_charge_out, vertex)` of the charge-augmented ITO
# automaton, together with the pass-through identity that fills idle sites and the caterpillar
# fusion-tree helpers (`bondcharges`, `vertexlabels`, `caterpillar_trees`). Prefixes agreeing on
# the full `ITOKey` share automaton state, so the automaton is block-diagonal in the bond charge.
# This is the alphabet the flat `ITOTermTable` / `irrep_mpo` per-sector bipartite + SVD pipeline
# consumes.
#
# Idle sites are filled with a distinguished pass-through identity symbol (trivial charge, index
# `n = 0`), not an enumerated `(c, n)` letter and not `one(A)`; it acts as `id(V)`. Supported term
# arity K ≥ 0; multi-channel (`GenericFusion`) coupling is deferred.
#
# Why a `Term` stores keys and not a fusion tree: for a caterpillar the keys *are* the tree —
# `bondcharges`/`vertexlabels` project one onto them and `_tree_from_bonds` (irrepalgebra.jl) inverts
# that. `total` alone would not do, since for K ≥ 3 non-abelian several channels reach the same total;
# and it rests on the shape being fixed, which is why `couple`'s `via` is deferred.

using TensorKit
using TensorKit: Sector, FusionTree, fusiontrees, unit
using .IrrepTensorOperators: IrrepOperator

"""
    passthrough(::Type{I}) where {I<:Sector}

The distinguished structural pass-through identity symbol: trivial charge `unit(I)` with the
sentinel index `n = 0`. Fills idle sites, injecting trivial charge and acting as `id(V)`. Not one
of the enumerated `(c, n)` alphabet letters (`n ≥ 1`) and not a type-level `one`.
"""
passthrough(::Type{I}) where {I <: Sector} = IrrepOperator{I}(unit(I), 0)

"""
    ispassthrough(op::IrrepOperator)

Whether `op` is the pass-through identity sentinel (index `n = 0`).
"""
ispassthrough(op::IrrepOperator) = op.n == 0

# Augmented trie key
# ------------------
"""
    ITOKey{I<:Sector}

Alphabet symbol of the charge-augmented automaton: `op` (the ITO letter or pass-through), `bond`
(the bond charge *out* of this site, i.e. the running fusion up to and including it), and `vertex`
(the fusion-vertex multiplicity label; `1` for multiplicity-free fusion). Prefixes agreeing on op
*and* bond *and* vertex share a trie node, so the trie is a direct sum of per-bond-sector FSMs.
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

# Caterpillar coupling tree + its per-site charges/labels
# -------------------------------------------------------
"""
    caterpillar_trees(charges::NTuple{K,I}, total::I) -> Vector{FusionTree{I,K}}

All left-nested caterpillar fusion trees of the per-site `charges` fusing to `total`, in
TensorKit's canonical order (finite, bounded — delegated to TensorKit `fusiontrees`).
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

_vlabel(v) = v isa Integer ? Int(v) : 1

"""
    vertexlabels(f::FusionTree{I,K})

Fusion-vertex multiplicity labels, one per active position (`1` at position 1, `f.vertices[k-1]`
after).
"""
function vertexlabels(f::FusionTree{I, K}) where {I, K}
    K == 0 && return Int[]
    return Int[k == 1 ? 1 : _vlabel(f.vertices[k - 1]) for k in 1:K]
end
