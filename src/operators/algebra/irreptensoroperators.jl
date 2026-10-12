module IrrepTensorOperators

export IrrepOperator

using OpSum: OperatorBasis
import OpSum: instantiate
using TensorKit
using VectorInterface

"""
    IrrepOperator{I<:Sector} <: OperatorBasis

Local irreducible tensor operator (ITO) alphabet element, labelled by an operator charge `c::I` (the
coupled leg) and a canonical integer `n` selecting which operator of that charge is meant. It is
interpreted relative to a physical space `V` supplied by context, not stored.

Materializes (via [`instantiate`](@ref)) to a `TensorMap` `O_{c,n} : V ← V ⊗ Vect[I](c => 1)`.

Index `1` of the trivial charge is the identity (see `passthrough`): element 1 of a
[`Basis`](@ref OpSum.Basis); in the matrix-unit alphabet the trivial-charge matrix units start at
`n = 2`. It instantiates to `isomorphism(V ← V ⊗ Vect[I](unit(I) => 1))`.
"""
struct IrrepOperator{I <: Sector} <: OperatorBasis
    c::I     # operator charge (the coupled leg)
    n::Int   # canonical index picking which operator of that charge
end

# There is one charge-`c` operator per scalar entry of the blocks of `O : V ← V ⊗ Vect[I](c => 1)`,
# so `dim(fuse(V ⊗ V'), c)` of them. `n` is the flat 1-based index into those entries in TensorKit's
# canonical block order (blocks by coupled sector, column-major within each); `instantiate` and
# `instances` are the two sides of this enumeration. The trivial charge is offset by one (`_muoffset`).
_muoffset(c::Sector) = c == unit(c) ? 1 : 0

"""
    instantiate(op::IrrepOperator, V::ElementarySpace)

Materialize the canonical ITO `TensorMap` `O_{c,n} : V ← V ⊗ Vect[I](c => 1)`.

The single nonzero entry (the `n`-th scalar over `blocks(O)`) is `1 / sqrt(dim(s))` for its block
sector `s`, so the alphabet is orthonormal under TensorKit's `inner`.
"""
function instantiate(op::IrrepOperator{I}, V::TensorKit.ElementarySpace) where {I <: Sector}
    sectortype(V) === I ||
        throw(ArgumentError("ITO sector $I incompatible with space sector $(sectortype(V))"))
    T = scalartype(IrrepOperator{I})
    op.n == 1 && op.c == unit(I) && return isomorphism(T, V ← V ⊗ Vect[I](op.c => 1))
    idx = op.n - _muoffset(op.c)
    idx >= 1 || throw(ArgumentError("index n=$(op.n) must be ≥ 1 for charge $(op.c)"))

    t = zeros(T, V ← V ⊗ Vect[I](op.c => 1))
    for (s, b) in blocks(t)
        if idx <= length(b)
            b[idx] = one(T) / sqrt(dim(s))
            return t
        end
        idx -= length(b)
    end
    throw(ArgumentError("index n=$(op.n) out of range for charge $(op.c) on space $V"))
end

"""
    instances(::Type{<:IrrepOperator}, V::ElementarySpace)

The matrix-unit alphabet spanning `End(V)`: `(c, n)` for every charge `c` in `fuse(V ⊗ V')`, one per
scalar entry. The identity is not among them: the trivial charge's letters are `n = 2, 3, …`.
"""
function Base.instances(::Type{<:IrrepOperator}, V::TensorKit.ElementarySpace)
    I = sectortype(V)
    W = fuse(V ⊗ V')
    return IrrepOperator{I}[IrrepOperator{I}(c, n + _muoffset(c)) for c in sectors(W) for n in 1:dim(W, c)]
end

# The `(c, n)` pair is the identity of the letter: order, equality and hashing reduce to it.
_key(x::IrrepOperator) = (x.c, x.n)
Base.isless(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I <: Sector} = isless(_key(x), _key(y))
Base.:(==)(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = _key(x) == _key(y)
Base.hash(x::IrrepOperator, h::UInt) = hash(_key(x), hash(:IrrepOperator, h))

# The scalar field is `sectorscalartype(I)` promoted to a complex float, so on-site and reduced
# coefficient arithmetic is uniformly complex (`ComplexF64` for SU(2), U(1), trivial).
VectorInterface.scalartype(::Type{IrrepOperator{I}}) where {I <: Sector} = complex(float(sectorscalartype(I)))
VectorInterface.scalartype(::Type{<:IrrepOperator}) = ComplexF64
Base.isreal(::Type{<:IrrepOperator}) = false

function VectorInterface.inner(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I <: Sector}
    T = scalartype(IrrepOperator{I})
    return x == y ? one(T) : zero(T)
end

function Base.show(io::IO, x::IrrepOperator)
    return print(io, "ITO(", x.c, ", n=", x.n, ")")
end

end
