# On-site operators: a sparse combination of alphabet letters, the element type of the reduced MPO's
# bond matrices. Stored as two parallel vectors rather than a `Dict` (one or two letters in practice,
# so a linear scan beats hashing and ordering is deterministic).
#
# The bare identity is the `passthrough` letter, so `scalarop(c, I)` is `c · passthrough` and
# consumers branch on `ispassthrough`. There is no symbolic on-site product (it would need 6j data);
# build it as a `TensorMap` and `project` it back.

using TensorKit: Sector, sectortype
using LinearAlgebra: LinearAlgebra
using VectorInterface: VectorInterface, inner

"""
    SiteOperator{I<:Sector}

An operator on a **single** site, not yet placed: a sparse combination of ITO alphabet letters,
`Σₖ coeffs[k] · letters[k]`.

Obtained from [`project`](@ref) into a [`Basis`](@ref), a builder such as [`spin_ops`](@ref), or
[`scalarop`](@ref), and combined with `+ - * /`. Place it on a site with `A[i]`, which produces a
[`Terms`](@ref) bag. The identity is the `passthrough` letter `(unit(I), 1)`.
"""
struct SiteOperator{I <: Sector}
    letters::Vector{IrrepOperator{I}}
    coeffs::Vector{ComplexF64}
    function SiteOperator{I}(letters::Vector{IrrepOperator{I}}, coeffs::Vector{ComplexF64}) where {I}
        @assert length(letters) == length(coeffs) "letters/coeffs length mismatch"
        return new{I}(letters, coeffs)
    end
end

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

# `pairs` is the accessor every consumer uses.
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

# the bare identity with unit coefficient
function Base.isone(op::SiteOperator)
    length(op) == 1 || return false
    return ispassthrough(only(op.letters)) && isone(only(op.coeffs))
end

function Base.:(==)(x::SiteOperator{I}, y::SiteOperator{I}) where {I}
    return x.letters == y.letters && x.coeffs == y.coeffs
end
Base.hash(x::SiteOperator, h::UInt) = hash(x.coeffs, hash(x.letters, hash(:SiteOperator, h)))

# `+` accumulates onto matching letters (what `increaseindex!` does when two bond entries share a slot).
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

# A bare letter promotes under all of the above (the sweep uses `key.op * coeff`).
Base.:*(x::IrrepOperator, a::Number) = SiteOperator(x) * a
Base.:*(a::Number, x::IrrepOperator) = SiteOperator(x) * a
Base.:/(x::IrrepOperator, a::Number) = SiteOperator(x) / a
Base.:+(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = SiteOperator(x) + SiteOperator(y)
Base.:-(x::IrrepOperator) = -SiteOperator(x)
Base.:-(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = SiteOperator(x) - SiteOperator(y)

VectorInterface.scale(x::SiteOperator, a::Number) = x * a
VectorInterface.add(x::SiteOperator{I}, y::SiteOperator{I}) where {I} = x + y

# `inner` likewise accepts a bare letter on either side.
VectorInterface.inner(x::IrrepOperator{I}, y::SiteOperator{I}) where {I} = inner(SiteOperator(x), y)
VectorInterface.inner(x::SiteOperator{I}, y::IrrepOperator{I}) where {I} = inner(x, SiteOperator(y))
VectorInterface.inner(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} =
    inner(SiteOperator(x), SiteOperator(y))

function VectorInterface.inner(x::SiteOperator{I}, y::SiteOperator{I}) where {I}
    # orthonormal alphabet: the plain coefficient overlap
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

# Drop letters whose coefficient cancelled to zero: a zero-weight edge would spend a bond index on nothing.
function prune(op::SiteOperator{I}) where {I}
    keep = findall(!iszero, op.coeffs)
    return SiteOperator{I}(op.letters[keep], op.coeffs[keep])
end

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
