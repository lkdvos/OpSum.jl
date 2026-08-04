# On-site operators: a sparse combination of alphabet letters
# ============================================================
# `OnsiteOp{I}` is what you get from `project`, `matrixunit`, `spin` and `scalarop`, and it is also
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
# constructed; `Prod`/`Pow` threw and `Kron`/`Fun` were unreachable. See research/onsite-products.md
# for what symbolic on-site products would actually require.

using TensorKit: Sector, sectortype
using LinearAlgebra: LinearAlgebra
using VectorInterface: VectorInterface, inner

"""
    OnsiteOp{I<:Sector}

An operator on a **single** site, not yet placed on the lattice: a sparse combination of ITO
alphabet letters, `Σₖ coeffs[k] · letters[k]`.

Obtained from [`project`](@ref), [`matrixunit`](@ref), [`spin`](@ref) or [`scalarop`](@ref), and
combined with ordinary arithmetic (`+ - * /`). Place it on the lattice with `A[i]`, which produces a
[`TermSum`](@ref).

The identity is represented as the `passthrough` sentinel letter, so `scalarop(c, I)` is `c` times
that letter rather than a distinct kind of object.
"""
struct OnsiteOp{I <: Sector}
    letters::Vector{IrrepOperator{I}}
    coeffs::Vector{ComplexF64}
    function OnsiteOp{I}(letters::Vector{IrrepOperator{I}}, coeffs::Vector{ComplexF64}) where {I}
        @assert length(letters) == length(coeffs) "letters/coeffs length mismatch"
        return new{I}(letters, coeffs)
    end
end

# Constructors
# ------------
OnsiteOp{I}() where {I <: Sector} = OnsiteOp{I}(IrrepOperator{I}[], ComplexF64[])
OnsiteOp(letters::Vector{IrrepOperator{I}}, coeffs::Vector{ComplexF64}) where {I} =
    OnsiteOp{I}(letters, coeffs)
OnsiteOp(op::IrrepOperator{I}) where {I} = OnsiteOp{I}([op], ComplexF64[1])
OnsiteOp{I}(c::Number) where {I <: Sector} =
    iszero(c) ? OnsiteOp{I}() : OnsiteOp{I}([passthrough(I)], ComplexF64[c])

Base.convert(::Type{OnsiteOp{I}}, op::IrrepOperator{I}) where {I} = OnsiteOp(op)
Base.convert(::Type{OnsiteOp{I}}, c::Number) where {I} = OnsiteOp{I}(c)

TensorKit.sectortype(::Type{OnsiteOp{I}}) where {I} = I
VectorInterface.scalartype(::Type{<:OnsiteOp}) = ComplexF64

# Container interface
# -------------------
# `pairs` is the accessor every consumer uses: it replaces the old `_local_terms`, which returned
# `(nothing, coeff)` for a scalar and `(letter, coeff)` otherwise. There is no `nothing` case now.
Base.pairs(op::OnsiteOp) = (l => c for (l, c) in zip(op.letters, op.coeffs))
Base.keys(op::OnsiteOp) = op.letters
Base.values(op::OnsiteOp) = op.coeffs
Base.length(op::OnsiteOp) = length(op.letters)
Base.isempty(op::OnsiteOp) = isempty(op.letters)
Base.iszero(op::OnsiteOp) = all(iszero, op.coeffs)

Base.zero(::Type{OnsiteOp{I}}) where {I} = OnsiteOp{I}()
Base.zero(::OnsiteOp{I}) where {I} = OnsiteOp{I}()
Base.one(::Type{OnsiteOp{I}}) where {I} = OnsiteOp{I}(1)
Base.one(::OnsiteOp{I}) where {I} = OnsiteOp{I}(1)

# `isone` is "the bare identity with unit coefficient", i.e. exactly the pass-through channel.
function Base.isone(op::OnsiteOp)
    length(op) == 1 || return false
    return ispassthrough(only(op.letters)) && isone(only(op.coeffs))
end

function Base.:(==)(x::OnsiteOp{I}, y::OnsiteOp{I}) where {I}
    return x.letters == y.letters && x.coeffs == y.coeffs
end
Base.hash(x::OnsiteOp, h::UInt) = hash(x.coeffs, hash(x.letters, hash(:OnsiteOp, h)))

# Arithmetic
# ----------
# `+` accumulates onto matching letters. This is the operation `increaseindex!` performs when two
# reduced bond entries land in the same `(row, col)` slot, which happens when two uncovered left
# vertices sharing an incoming link fold into the same covered-right bond index.
function Base.:+(x::OnsiteOp{I}, y::OnsiteOp{I}) where {I}
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
    return OnsiteOp{I}(letters, coeffs)
end

Base.:-(x::OnsiteOp{I}) where {I} = OnsiteOp{I}(copy(x.letters), -x.coeffs)
Base.:-(x::OnsiteOp{I}, y::OnsiteOp{I}) where {I} = x + (-y)

Base.:*(x::OnsiteOp{I}, a::Number) where {I} = OnsiteOp{I}(copy(x.letters), x.coeffs .* a)
Base.:*(a::Number, x::OnsiteOp) = x * a
Base.:/(x::OnsiteOp, a::Number) = x * inv(a)

# A bare letter times a scalar promotes to an `OnsiteOp`. The per-bond sweep leans on this when it
# weights a letter by its reduced coefficient (`key.op * coeff`).
Base.:*(x::IrrepOperator, a::Number) = OnsiteOp(x) * a
Base.:*(a::Number, x::IrrepOperator) = OnsiteOp(x) * a
Base.:+(x::IrrepOperator{I}, y::IrrepOperator{I}) where {I} = OnsiteOp(x) + OnsiteOp(y)

VectorInterface.scale(x::OnsiteOp, a::Number) = x * a
VectorInterface.add(x::OnsiteOp{I}, y::OnsiteOp{I}) where {I} = x + y

function VectorInterface.inner(x::OnsiteOp{I}, y::OnsiteOp{I}) where {I}
    # the alphabet is orthonormal, so the inner product is the plain coefficient overlap
    result = zero(ComplexF64)
    for (lx, cx) in zip(x.letters, x.coeffs), (ly, cy) in zip(y.letters, y.coeffs)
        lx == ly && (result += conj(cx) * cy)
    end
    return result
end
LinearAlgebra.norm(x::OnsiteOp) = sqrt(abs(inner(x, x)))

function Base.isapprox(x::OnsiteOp{I}, y::OnsiteOp{I}; kwargs...) where {I}
    return isapprox(norm(x - y), 0; atol = max(norm(x), norm(y)) * 1.0e-12, kwargs...)
end

# Drop letters whose coefficient has cancelled to zero. Bond entries must not carry explicit zeros:
# the cover treats a zero-weight edge as absent, so keeping one would spend a bond index on nothing.
function prune(op::OnsiteOp{I}) where {I}
    keep = findall(!iszero, op.coeffs)
    return OnsiteOp{I}(op.letters[keep], op.coeffs[keep])
end

# Show
# ----
function Base.show(io::IO, op::OnsiteOp{I}) where {I}
    isempty(op) && return print(io, "OnsiteOp{", I, "}(0)")
    print(io, "OnsiteOp{", I, "}(")
    for (k, (l, c)) in enumerate(zip(op.letters, op.coeffs))
        k > 1 && print(io, " + ")
        isone(c) || print(io, c, "*")
        show(io, l)
    end
    return print(io, ")")
end
