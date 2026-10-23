# On-site operators: a sparse combination of alphabet letters, the element type of the reduced MPO's
# bond matrices. Two parallel vectors, not a `Dict`: one or two letters in practice, deterministic order.

using TensorKit: Sector, sectortype
using LinearAlgebra: LinearAlgebra
using VectorInterface: VectorInterface, inner

"""
    SiteOperator{I<:Sector}

An unplaced single-site operator `Σₖ coeffs[k] · letters[k]`, from [`project`](@ref), [`spin_ops`](@ref)
and friends or [`scalarop`](@ref); combine with `+ - * /` and place with `A[i]` (a [`Terms`](@ref) bag).
"""
struct SiteOperator{I <: Sector}
    letters::Vector{Letter{I}}
    coeffs::Vector{ComplexF64}
    function SiteOperator{I}(letters::Vector{Letter{I}}, coeffs::Vector{ComplexF64}) where {I}
        @assert length(letters) == length(coeffs) "letters/coeffs length mismatch"
        return new{I}(letters, coeffs)
    end
end

SiteOperator{I}() where {I <: Sector} = SiteOperator{I}(Letter{I}[], ComplexF64[])
SiteOperator(letters::Vector{Letter{I}}, coeffs::Vector{ComplexF64}) where {I} =
    SiteOperator{I}(letters, coeffs)
SiteOperator(op::Letter{I}) where {I} = SiteOperator{I}([op], ComplexF64[1])
SiteOperator{I}(c::Number) where {I <: Sector} =
    iszero(c) ? SiteOperator{I}() : SiteOperator{I}([passthrough(I)], ComplexF64[c])

Base.convert(::Type{SiteOperator{I}}, op::Letter{I}) where {I} = SiteOperator(op)
Base.convert(::Type{SiteOperator{I}}, c::Number) where {I} = SiteOperator{I}(c)

TensorKit.sectortype(::Type{SiteOperator{I}}) where {I} = I
VectorInterface.scalartype(::Type{<:SiteOperator}) = ComplexF64

Base.pairs(op::SiteOperator) = (l => c for (l, c) in zip(op.letters, op.coeffs))
Base.keys(op::SiteOperator) = op.letters
Base.values(op::SiteOperator) = op.coeffs
Base.length(op::SiteOperator) = length(op.letters)
Base.isempty(op::SiteOperator) = isempty(op.letters)
Base.iszero(op::SiteOperator) = all(iszero, op.coeffs)

Base.zero(::Union{SiteOperator{I}, Type{SiteOperator{I}}}) where {I} = SiteOperator{I}()
Base.one(::Union{SiteOperator{I}, Type{SiteOperator{I}}}) where {I} = SiteOperator{I}(1)

Base.isone(op::SiteOperator) = length(op) == 1 && ispassthrough(only(op.letters)) && isone(only(op.coeffs))

Base.:(==)(x::SiteOperator{I}, y::SiteOperator{I}) where {I} = x.letters == y.letters && x.coeffs == y.coeffs
Base.hash(x::SiteOperator, h::UInt) = hash(x.coeffs, hash(x.letters, hash(:SiteOperator, h)))

# accumulates onto matching letters (two bond entries sharing a slot)
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

# A bare letter times a scalar promotes (the sweep uses `key.letter * coeff`).
Base.:*(x::Letter, a::Number) = SiteOperator(x) * a
Base.:*(a::Number, x::Letter) = SiteOperator(x) * a

VectorInterface.scale(x::SiteOperator, a::Number) = x * a

# orthonormal alphabet: the plain coefficient overlap; a bare letter is accepted on either side
function VectorInterface.inner(
        x::Union{Letter{I}, SiteOperator{I}}, y::Union{Letter{I}, SiteOperator{I}}
    ) where {I}
    x, y = convert(SiteOperator{I}, x), convert(SiteOperator{I}, y)
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

# drop letters whose coefficient cancelled (a zero-weight edge would spend a bond index)
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
