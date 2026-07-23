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

The sentinel index `n = 0` (with `c = unit(I)`) is reserved for the Phase-3 pass-through
identity (see `passthrough`): it is not an enumerated alphabet letter (`instances` yields
`n ≥ 1`) and instantiates to the structural identity `id(V)`.
"""
struct IrrepOperator{I <: Sector} <: OperatorBasis
    c::I     # operator charge (the coupled leg)
    n::Int   # canonical index picking which operator of that charge
end

# Canonical (c)-channel enumeration
# ---------------------------------
# The operators of charge `c` on `V` are the reduced coefficients of a tensor `O : V ← V ⊗ Cc`
# with `Cc = Vect[I](c => 1)`. There is exactly one such coefficient per scalar entry of the
# block matrices of `O`, so the number of charge-`c` operators equals `dim(fuse(V ⊗ V'), c)` (and
# the total over all charges is `dim(V ⊗ V')`). `n` is the flat 1-based index into those entries,
# in TensorKit's canonical block order (blocks by coupled sector, column-major within each block);
# `instantiate` and `instances` are the two sides of that single enumeration.

# Materialization
# ---------------
"""
    instantiate(op::IrrepOperator, V::ElementarySpace)

Materialize the canonical ITO `TensorMap` `O_{c,n} : V ← V ⊗ Vect[I](c => 1)`.

The single nonzero reduced entry (selected by the canonical `n`-ordering: the `n`-th scalar over
`blocks(O)`) is set to `1 / sqrt(dim(s))`, where `s` is that entry's coupled/block sector. This
normalizes the alphabet to be orthonormal under TensorKit's qdim-weighted `inner` (so
`inner(O, O) = dim(s)·|entry|² = 1`).
"""
function instantiate(op::IrrepOperator{I}, V::TensorKit.ElementarySpace) where {I <: Sector}
    sectortype(V) === I ||
        throw(ArgumentError("ITO sector $I incompatible with space sector $(sectortype(V))"))
    # pass-through identity sentinel (Phase 3): trivial charge, n == 0, acts as id(V)
    T = scalartype(IrrepOperator{I})

    op.n == 0 && op.c == unit(I) && return id(T, V)
    op.n >= 1 || throw(ArgumentError("index n=$(op.n) must be ≥ 1 for charge $(op.c)"))

    t = zeros(T, V ← V ⊗ Vect[I](op.c => 1))
    idx = op.n
    for (s, b) in blocks(t)
        if idx <= length(b)
            b[idx] = one(T) / sqrt(dim(s))
            return t
        end
        idx -= length(b)
    end
    throw(ArgumentError("index n=$(op.n) out of range for charge $(op.c) on space $V"))
end

# Space-aware interface
# ---------------------
"""
    instances(::Type{<:IrrepOperator}, V::ElementarySpace)

Return the full ITO alphabet spanning `End(V)`: all `(c, n)` for every charge `c` appearing in
`fuse(V ⊗ V')`, with the complete `n` range `1:dim(fuse(V ⊗ V'), c)` (canonical block ordering).
"""
function Base.instances(::Type{<:IrrepOperator}, V::TensorKit.ElementarySpace)
    I = sectortype(V)
    W = fuse(V ⊗ V')
    return IrrepOperator{I}[IrrepOperator{I}(c, n) for c in sectors(W) for n in 1:dim(W, c) ]
end

# Ordering / hashing (sorting Us, Dictionary keys)
# ------------------------------------------------
# The `(c, n)` pair is the identity of the letter: order, equality and hashing all reduce to it.
# Tuple `isless` orders by TensorKit's canonical sector `isless` first, tie-broken by `n`.
_key(x::IrrepOperator) = (x.c, x.n)
Base.isless(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I <: Sector} = isless(_key(x), _key(y))
Base.:(==)(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = _key(x) == _key(y)
Base.hash(x::IrrepOperator, h::UInt) = hash(_key(x), hash(:IrrepOperator, h))

# Scalars / reality
# -----------------
# The scalar field follows the sector's topological data via `sectorscalartype(I)`, promoted to a
# *complex* floating field: real/integer topological data (e.g. SU(2), U(1), trivial) gives
# `ComplexF64`, matching the coefficient type the rest of the term algebra / MPO pipeline works in;
# genuinely anyonic (complex F-symbol) sectors keep their complex field. Staying complex avoids
# `LocalOp{Float64}`/`LocalOp{ComplexF64}` mismatches in the symbolic algebra. The bare
# (sector-free) type falls back to `ComplexF64`.
VectorInterface.scalartype(::Type{IrrepOperator{I}}) where {I <: Sector} = complex(float(sectorscalartype(I)))
VectorInterface.scalartype(::Type{<:IrrepOperator}) = ComplexF64
Base.isreal(::Type{<:IrrepOperator}) = false

function VectorInterface.inner(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I <: Sector}
    T = scalartype(IrrepOperator{I})
    return x == y ? one(T) : zero(T)
end

# Display
# -------
function Base.show(io::IO, x::IrrepOperator)
    return print(io, "ITO(", x.c, ", n=", x.n, ")")
end

end
