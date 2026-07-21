module IrrepTensorOperators

export IrrepOperator

using OpSum: OperatorBasis, LocalOp
import OpSum: instantiate
using TensorKit
using VectorInterface

# Element type
# ------------
"""
    IrrepOperator{I<:Sector} <: OperatorBasis

Local irreducible tensor operator (ITO) alphabet element, labelled by an operator charge
`c::I` (the coupled leg) and a canonical integer `n` selecting which operator of that charge
is meant. The element is interpreted **relative to a physical space `V`** which is supplied by
context (like the `sites`/axes argument of `instantiate`), not stored in the struct.

Materializes (via [`instantiate`](@ref)) to a TensorKit `TensorMap`

    O_{c,n} :  V  ←  V ⊗ Vect[I](c => 1)

i.e. an operator `V←V` carrying a dangling charge-`c` leg of degeneracy 1.
"""
struct IrrepOperator{I <: Sector} <: OperatorBasis
    c::I     # operator charge (the coupled leg)
    n::Int   # canonical index picking which operator of that charge
end

# Canonical (c)-channel enumeration
# ---------------------------------
# Single source of truth shared by `instantiate` and `instances`.
#
# For a fixed charge `c` and physical space `V`, the operators of charge `c` are enumerated
# deterministically as `(f₁, f₂, row, col)` tuples, ordered over:
#   1. the coupled/block sector `j′ ∈ sectors(V)` (sorted),
#   2. the domain fusion tree `f₂` of `(j, c) → j′` for `j ∈ sectors(V)` (sorted, inner tree
#      iteration deterministic),
#   3. the codomain degeneracy index `row ∈ 1:dim(V, j′)` (outer),
#   4. the domain degeneracy index `col ∈ 1:dim(V, j)` (inner).
# `n` is the flat 1-based index into that ordered list. `f₁` is the (unique) codomain splitting
# tree of the single leg `j′`.
function _channels(::Type{<:IrrepOperator}, c::I, V::TensorKit.ElementarySpace) where {I <: Sector}
    sectortype(V) === I ||
        throw(ArgumentError("charge sector $I incompatible with space sector $(sectortype(V))"))
    table = Tuple{FusionTree, FusionTree, Int, Int}[]
    for jp in sectors(V)
        f1 = only(fusiontrees((jp,), jp, (false,)))
        for j in sectors(V)
            for f2 in fusiontrees((j, c), jp, (false, false))
                for row in 1:dim(V, jp), col in 1:dim(V, j)
                    push!(table, (f1, f2, row, col))
                end
            end
        end
    end
    return table
end

# all charges `c` appearing in `End(V) ≅ V ⊗ V*`, i.e. `c ∈ dual(j) ⊗ j′` for `j, j′ ∈ V`.
function _charges(V::TensorKit.ElementarySpace)
    I = sectortype(V)
    cs = I[c for j in sectors(V) for jp in sectors(V) for c in (dual(j) ⊗ jp)]
    return sort!(unique!(cs))
end

# Materialization
# ---------------
"""
    instantiate(op::IrrepOperator, V::ElementarySpace)

Materialize the canonical ITO `TensorMap` `O_{c,n} : V ← V ⊗ Vect[I](c => 1)`.

The single nonzero reduced entry (selected by the canonical `n`-ordering, see `_channels`) is
set to `1 / sqrt(dim(j′))`, where `j′` is the coupled/block sector. This normalizes the alphabet
to be orthonormal under TensorKit's qdim-weighted `inner` (so `inner(O, O) = dim(j′)·|entry|² = 1`).
"""
function instantiate(op::IrrepOperator{I}, V::TensorKit.ElementarySpace) where {I <: Sector}
    sectortype(V) === I ||
        throw(ArgumentError("ITO sector $I incompatible with space sector $(sectortype(V))"))
    table = _channels(IrrepOperator, op.c, V)
    1 <= op.n <= length(table) ||
        throw(ArgumentError("index n=$(op.n) out of range 1:$(length(table)) for charge $(op.c)"))
    f1, f2, row, col = table[op.n]

    T = scalartype(IrrepOperator)
    Cc = Vect[I](op.c => 1)
    t = zeros(T, V ← V ⊗ Cc)
    t[f1, f2][row, col, 1] = one(T) / sqrt(dim(f1.coupled))
    return t
end

# Space-aware interface
# ---------------------
"""
    instances(::Type{<:IrrepOperator}, V::ElementarySpace)

Return the full ITO alphabet spanning `End(V)`: all `(c, n)` for every charge `c` appearing in
`dual(V) ⊗ V`, with their complete `n` ranges (canonical `_channels` ordering).
"""
function Base.instances(::Type{<:IrrepOperator}, V::TensorKit.ElementarySpace)
    I = sectortype(V)
    ops = IrrepOperator{I}[]
    for c in _charges(V)
        for n in 1:length(_channels(IrrepOperator, c, V))
            push!(ops, IrrepOperator{I}(c, n))
        end
    end
    return ops
end

"""
    one(::Type{<:IrrepOperator}, V::ElementarySpace)

The `c = unit(I)` (trivial-charge) identity ITO. Its dense form is the normalized identity
`id(V) / sqrt(dim(j′))` **only when the trivial-charge sector of `End(V)` is one-dimensional**
(e.g. a single SU(2) spin). For spaces with degeneracy or multiple sectors (e.g. `ℂ^2`, U(1))
the true identity is a linear combination of trivial-charge ITOs; this accessor then returns the
first trivial-charge matrix-unit ITO. A robust composite-identity accessor is deferred to Phase 2.
"""
function Base.one(::Type{<:IrrepOperator}, V::TensorKit.ElementarySpace)
    I = sectortype(V)
    return IrrepOperator{I}(unit(I), 1)
end

# Ordering / hashing (Trie / Dictionary keys)
# -------------------------------------------
# Order by TensorKit's canonical sector `isless` first, tie-broken by the canonical index `n`.
function Base.isless(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I <: Sector}
    return x.c == y.c ? isless(x.n, y.n) : isless(x.c, y.c)
end
Base.:(==)(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = x.c == y.c && x.n == y.n
Base.hash(x::IrrepOperator, h::UInt) = hash(x.n, hash(x.c, hash(:IrrepOperator, h)))

# Scalars / reality
# -----------------
# Reality decision: default to `ComplexF64`/`isreal = false` for now (safest given CG/F-symbol
# phases). A real convention for SU(2) can be revisited later.
VectorInterface.scalartype(::Type{<:IrrepOperator}) = ComplexF64
Base.isreal(::Type{<:IrrepOperator}) = false

# Since the alphabet is orthonormal by construction, the symbolic inner product collapses to
# equality. Validated in the tests against `inner(instantiate(x, V), instantiate(y, V))`.
function VectorInterface.inner(x::IrrepOperator, y::IrrepOperator)
    T = scalartype(IrrepOperator)
    return x == y ? one(T) : zero(T)
end

# Display
# -------
function Base.show(io::IO, x::IrrepOperator)
    return print(io, "ITO(", x.c, ", n=", x.n, ")")
end

end
