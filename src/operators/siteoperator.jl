# On-site operators: a sparse combination of alphabet letters
# ============================================================
# `SiteOperator{I}` is what you get from `project`, `matrixunit`, `spin` and `scalarop`, and it is also
# the element type of the reduced MPO's bond matrices. It is a small ordered map
# `IrrepOperator{I} => ComplexF64`, nothing more.
#
# The bare identity is *not* a special field: it is the `passthrough` sentinel letter
# (`IrrepOperator{I}(unit(I), 0)`), which already means "acts as `id(V)`, carries no charge" and
# already instantiates to `id(V)`. So `scalarop(c, I)` is literally `c · passthrough`, and every
# consumer branches on `ispassthrough` — a predicate that has to exist anyway for the idle sites of
# the sweep — instead of on a separate scalar slot.
#
# Entries are two parallel vectors rather than a `Dict`: an on-site operator has one or two letters
# in every realistic case (`Sz` has two, `spin` has one), and linear scan over a two-element vector
# beats hashing while keeping the ordering deterministic.
#
# This replaces the `LocalOp{T,A}` sum type over `Sum`/`Prod`/`Pow`/`Kron`/`Fun`. Only `Sum` was ever
# constructed; `Prod`/`Pow` threw and `Kron`/`Fun` were unreachable. A symbolic on-site *product*
# would need structure constants for the alphabet — for ITOs, the decomposition of a product of two
# irreducible tensor operators on one site, which is 6j data rather than a rewrite of this type — so
# the supported route is to build the product as a `TensorMap` and `project` it back.

using TensorKit: Sector, sectortype
using LinearAlgebra: LinearAlgebra
using VectorInterface: VectorInterface, inner

"""
    SiteOperator{I<:Sector}

An operator on a **single** site, not yet placed on the lattice: a sparse combination of ITO
alphabet letters, `Σₖ coeffs[k] · letters[k]`.

Obtained from [`project`](@ref), [`matrixunit`](@ref), [`spin`](@ref) or [`scalarop`](@ref), and
combined with ordinary arithmetic (`+ - * /`). Place it on the lattice with `A[i]`, which produces a
[`TermSum`](@ref).

The identity is represented as the `passthrough` sentinel letter, so `scalarop(c, I)` is `c` times
that letter rather than a distinct kind of object.
"""
struct SiteOperator{I <: Sector}
    letters::Vector{IrrepOperator{I}}
    coeffs::Vector{ComplexF64}
    function SiteOperator{I}(letters::Vector{IrrepOperator{I}}, coeffs::Vector{ComplexF64}) where {I}
        @assert length(letters) == length(coeffs) "letters/coeffs length mismatch"
        return new{I}(letters, coeffs)
    end
end

# Constructors
# ------------
SiteOperator{I}() where {I <: Sector} = SiteOperator{I}(IrrepOperator{I}[], ComplexF64[])
SiteOperator(letters::Vector{IrrepOperator{I}}, coeffs::Vector{ComplexF64}) where {I} =
    SiteOperator{I}(letters, coeffs)
SiteOperator(op::IrrepOperator{I}) where {I} = SiteOperator{I}([op], ComplexF64[1])
SiteOperator{I}(c::Number) where {I <: Sector} =
    iszero(c) ? SiteOperator{I}() : SiteOperator{I}([passthrough(I)], ComplexF64[c])

Base.convert(::Type{SiteOperator{I}}, op::IrrepOperator{I}) where {I} = SiteOperator(op)
Base.convert(::Type{SiteOperator{I}}, c::Number) where {I} = SiteOperator{I}(c)

TensorKit.sectortype(::Type{SiteOperator{I}}) where {I} = I
VectorInterface.scalartype(::Type{<:SiteOperator}) = ComplexF64

# Container interface
# -------------------
# `pairs` is the accessor every consumer uses: it replaces the old `_local_terms`, which returned
# `(nothing, coeff)` for a scalar and `(letter, coeff)` otherwise. There is no `nothing` case now.
Base.pairs(op::SiteOperator) = (l => c for (l, c) in zip(op.letters, op.coeffs))
Base.keys(op::SiteOperator) = op.letters
Base.values(op::SiteOperator) = op.coeffs
Base.length(op::SiteOperator) = length(op.letters)
Base.isempty(op::SiteOperator) = isempty(op.letters)
Base.iszero(op::SiteOperator) = all(iszero, op.coeffs)

Base.zero(::Type{SiteOperator{I}}) where {I} = SiteOperator{I}()
Base.zero(::SiteOperator{I}) where {I} = SiteOperator{I}()
Base.one(::Type{SiteOperator{I}}) where {I} = SiteOperator{I}(1)
Base.one(::SiteOperator{I}) where {I} = SiteOperator{I}(1)

# `isone` is "the bare identity with unit coefficient", i.e. exactly the pass-through channel.
function Base.isone(op::SiteOperator)
    length(op) == 1 || return false
    return ispassthrough(only(op.letters)) && isone(only(op.coeffs))
end

function Base.:(==)(x::SiteOperator{I}, y::SiteOperator{I}) where {I}
    return x.letters == y.letters && x.coeffs == y.coeffs
end
Base.hash(x::SiteOperator, h::UInt) = hash(x.coeffs, hash(x.letters, hash(:SiteOperator, h)))

# Arithmetic
# ----------
# `+` accumulates onto matching letters. This is the operation `increaseindex!` performs when two
# reduced bond entries land in the same `(row, col)` slot, which happens when two uncovered left
# vertices sharing an incoming link fold into the same covered-right bond index.
function Base.:+(x::SiteOperator{I}, y::SiteOperator{I}) where {I}
    letters = copy(x.letters)
    coeffs = copy(x.coeffs)
    for (l, c) in zip(y.letters, y.coeffs)
        k = findfirst(==(l), letters)
        if k === nothing
            push!(letters, l)
            push!(coeffs, c)
        else
            coeffs[k] += c
        end
    end
    return SiteOperator{I}(letters, coeffs)
end

Base.:-(x::SiteOperator{I}) where {I} = SiteOperator{I}(copy(x.letters), -x.coeffs)
Base.:-(x::SiteOperator{I}, y::SiteOperator{I}) where {I} = x + (-y)

Base.:*(x::SiteOperator{I}, a::Number) where {I} = SiteOperator{I}(copy(x.letters), x.coeffs .* a)
Base.:*(a::Number, x::SiteOperator) = x * a
Base.:/(x::SiteOperator, a::Number) = x * inv(a)

# A bare letter times a scalar promotes to an `SiteOperator`. The per-bond sweep leans on this when it
# weights a letter by its reduced coefficient (`key.op * coeff`).
Base.:*(x::IrrepOperator, a::Number) = SiteOperator(x) * a
Base.:*(a::Number, x::IrrepOperator) = SiteOperator(x) * a
Base.:+(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = SiteOperator(x) + SiteOperator(y)

VectorInterface.scale(x::SiteOperator, a::Number) = x * a
VectorInterface.add(x::SiteOperator{I}, y::SiteOperator{I}) where {I} = x + y

function VectorInterface.inner(x::SiteOperator{I}, y::SiteOperator{I}) where {I}
    # the alphabet is orthonormal, so the inner product is the plain coefficient overlap
    result = zero(ComplexF64)
    for (lx, cx) in zip(x.letters, x.coeffs), (ly, cy) in zip(y.letters, y.coeffs)
        lx == ly && (result += conj(cx) * cy)
    end
    return result
end
LinearAlgebra.norm(x::SiteOperator) = sqrt(abs(inner(x, x)))

function Base.isapprox(x::SiteOperator{I}, y::SiteOperator{I}; kwargs...) where {I}
    return isapprox(norm(x - y), 0; atol = max(norm(x), norm(y)) * 1.0e-12, kwargs...)
end

# Drop letters whose coefficient has cancelled to zero. Bond entries must not carry explicit zeros:
# the cover treats a zero-weight edge as absent, so keeping one would spend a bond index on nothing.
function prune(op::SiteOperator{I}) where {I}
    keep = findall(!iszero, op.coeffs)
    return SiteOperator{I}(op.letters[keep], op.coeffs[keep])
end

# Show
# ----
function Base.show(io::IO, op::SiteOperator{I}) where {I}
    isempty(op) && return print(io, "SiteOperator{", I, "}(0)")
    print(io, "SiteOperator{", I, "}(")
    for (k, (l, c)) in enumerate(zip(op.letters, op.coeffs))
        k > 1 && print(io, " + ")
        isone(c) || print(io, c, "*")
        show(io, l)
    end
    return print(io, ")")
end
