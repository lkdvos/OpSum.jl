# The letter: operator charge and index into the site's `Basis`, plus the `Factor` (letter, running bond
# charge) a `Term` stores per site. Letters are interpreted by `matrixunit` (on a bare space) or `instantiate` (in a `Basis`) only.

using TensorKit
using TensorKit: Sector, FusionTree, fusiontrees, unit

"""
    Letter{I<:Sector}

An irreducible tensor operator letter: operator charge `c::I` and index `n` among the operators of
that charge, interpreted relative to a site's operator [`Basis`](@ref OpSum.Basis): [`instantiate`](@ref)
gives the element `O_{c,n} : V ← V ⊗ Vect[I](c => 1)`. `(unit, 1)` is the identity, element 1 of a `Basis`.
"""
struct Letter{I <: Sector}
    c::I     # operator charge (the coupled leg)
    n::Int   # canonical index picking which operator of that charge
end

# `n` is the flat 1-based index over the scalar entries of the blocks of `O : V ← V ⊗ Vect[I](c => 1)`
# in TensorKit's block order (`dim(fuse(V ⊗ V'), c)` of them); `matrixunit` and `matrixunits` are the two
# sides. The trivial charge is offset by one: element 1 is the identity.
_muoffset(c::Sector) = c == unit(c) ? 1 : 0

# the single nonzero entry is `1 / sqrt(dim(s))` in block `s`, so the alphabet is `inner`-orthonormal
"""
    matrixunit(op::Letter, V::ElementarySpace)

The matrix-unit tensor `V ← V ⊗ Vect[I](c => 1)` of the letter `op` on the space `V`: the single nonzero
entry is `1 / sqrt(dim(s))` in block `s`. A [`Basis`](@ref OpSum.Basis) interprets letters through
[`instantiate`](@ref) instead.
"""
function matrixunit(op::Letter{I}, V::TensorKit.ElementarySpace) where {I <: Sector}
    sectortype(V) === I ||
        throw(ArgumentError("Letter sector $I incompatible with space sector $(sectortype(V))"))
    T = scalartype(Letter{I})
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
function matrixunits(V::TensorKit.ElementarySpace)
    I = sectortype(V)
    W = fuse(V ⊗ V')
    return Letter{I}[Letter{I}(c, n + _muoffset(c)) for c in sectors(W) for n in 1:dim(W, c)]
end

# `==` is the struct default; `hash` is field-wise (the default goes through `objectid`, slow for product sectors)
Base.hash(x::Letter, h::UInt) = hash(x.n, hash(x.c, hash(:Letter, h)))
Base.isless(x::Letter{I}, y::Letter{I}) where {I <: Sector} = isless((x.c, x.n), (y.c, y.n))

VectorInterface.scalartype(::Type{<:Letter}) = ComplexF64

function VectorInterface.inner(x::Letter{I}, y::Letter{I}) where {I <: Sector}
    T = scalartype(Letter{I})
    return x == y ? one(T) : zero(T)
end

function Base.show(io::IO, x::Letter)
    return print(io, "Letter(", x.c, ", n=", x.n, ")")
end

# sector type of the letter alphabet
TensorKit.sectortype(::Type{Letter{I}}) where {I} = I

# `Factor = (letter, bond_charge_out)`. `Term` stores keys, not a fusion tree: for a caterpillar the
# keys *are* the tree (`bondcharges` projects onto them, `tree_from_bonds` inverts it). The total alone
# would not do: for K ≥ 3 non-abelian several channels reach the same total.

# the identity letter, element 1 of every `Basis`; fills idle sites, and placement drops it
passthrough(::Type{I}) where {I <: Sector} = Letter{I}(unit(I), 1)

ispassthrough(op::Letter) = op == passthrough(sectortype(op))

# `bond` is the running fusion charge out of the site (multiplicity-free fusion: no vertex label)
struct Factor{I <: Sector}
    letter::Letter{I}
    bond::I
end

Base.hash(x::Factor, h::UInt) = hash(x.bond, hash(x.letter, hash(:Factor, h)))
Base.isless(x::Factor{I}, y::Factor{I}) where {I} = isless((x.letter, x.bond), (y.letter, y.bond))

function Base.show(io::IO, k::Factor)
    return print(io, "Factor(", k.letter, ", bond=", k.bond, ")")
end

# A caterpillar's factors, read as its letters, their charges, its running bond charges and its total
# (the last bond; the unit for no factors).
letters(factors::Vector{Factor{I}}) where {I} = Letter{I}[f.letter for f in factors]
lettercharges(factors::Vector{Factor{I}}) where {I} = I[f.letter.c for f in factors]
bondcharges(factors::Vector{Factor{I}}) where {I} = I[f.bond for f in factors]
total(factors::Vector{Factor{I}}) where {I} = isempty(factors) ? unit(I) : last(factors).bond

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
