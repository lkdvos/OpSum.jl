module IrrepTensorOperators

export IrrepOperator

import OpSum: instantiate
using TensorKit
using VectorInterface

"""
    IrrepOperator{I<:Sector}

An irreducible tensor operator letter: operator charge `c::I` and index `n` among the operators of
that charge, interpreted relative to a physical space supplied by context. [`instantiate`](@ref)
gives `O_{c,n} : V ← V ⊗ Vect[I](c => 1)`. `(unit, 1)` is the identity, element 1 of a [`Basis`](@ref OpSum.Basis).
"""
struct IrrepOperator{I <: Sector}
    c::I     # operator charge (the coupled leg)
    n::Int   # canonical index picking which operator of that charge
end

# `n` is the flat 1-based index over the scalar entries of the blocks of `O : V ← V ⊗ Vect[I](c => 1)`
# in TensorKit's block order (`dim(fuse(V ⊗ V'), c)` of them); `instantiate` and `instances` are the two
# sides. The trivial charge is offset by one: element 1 is the identity.
_muoffset(c::Sector) = c == unit(c) ? 1 : 0

# the single nonzero entry is `1 / sqrt(dim(s))` in block `s`, so the alphabet is `inner`-orthonormal
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

# the matrix-unit alphabet spanning `End(V)` (the identity is not among them: trivial letters start at n = 2)
function Base.instances(::Type{<:IrrepOperator}, V::TensorKit.ElementarySpace)
    I = sectortype(V)
    W = fuse(V ⊗ V')
    return IrrepOperator{I}[IrrepOperator{I}(c, n + _muoffset(c)) for c in sectors(W) for n in 1:dim(W, c)]
end

_key(x::IrrepOperator) = (x.c, x.n)
Base.isless(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I <: Sector} = isless(_key(x), _key(y))
Base.:(==)(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = _key(x) == _key(y)
Base.hash(x::IrrepOperator, h::UInt) = hash(_key(x), hash(:IrrepOperator, h))

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

using .IrrepTensorOperators: IrrepOperator
import TensorKit: sectortype
# sector type of the ITO alphabet
sectortype(::Type{IrrepOperator{I}}) where {I} = I

# `ITOKey = (op, bond_charge_out, vertex)`. `Term` stores keys, not a fusion tree: for a caterpillar the
# keys *are* the tree (`bondcharges` projects onto them, `_tree_from_bonds` inverts it). The total alone
# would not do: for K ≥ 3 non-abelian several channels reach the same total.

using TensorKit
using TensorKit: Sector, FusionTree, fusiontrees, unit

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
