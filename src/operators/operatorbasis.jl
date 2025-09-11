"""
    abstract type OperatorBasis

Abstract supertype for all operator basis elements.
"""
abstract type OperatorBasis end

function namemap end

function Base.instances(::Type{O}) where {O<:OperatorBasis}
    maps = namemap(O)
    return O.(keys(maps))
end

function Base.show(io::IO, x::OperatorBasis)
    maps = namemap(typeof(x))
    return show(io, maps[Integer(x)])
end

# Instantiation
# -------------
"""
    instantiate(b::OperatorBasis, ::Type{T}, axes) where {T <: Number}

Instantiate an `AbstractArray{T}` instance for the given basis element.
"""
function instantiate end

# Expansion
# ---------
function LinearAlgebra.dot(x::OperatorBasis, y::AbstractArray)
    N = ndims(y)
    @assert iseven(N) "operators have an equal number of in and out legs"
    for i in 1:(N ÷ 2)
        @assert axes(y, i) == axes(y, i + (N ÷ 2)) "operators must have equal in and out legs"
    end

    yax = map(Base.Fix1(axes, y), 1:(N÷2))
    T = VectorInterface.promote_inner(x, y)
    x′ = instantiate(x, T, yax)
    return LinearAlgebra.dot(x′, y)
end

"""
    project(::Type{T}, operator) where {T<:OperatorBasis}

Rewrite the given operator by expanding it into the given basis set.
"""
function project(::Type{T}, operator) where {T <: OperatorBasis}
    expanded_operator = sum(instances(T)) do basis_element
        c = LinearAlgebra.dot(basis_element, operator)
        return c * basis_element
    end
    # norm(expanded_operator) ≈ norm(operator) ||
    #     @warn "Expansion did not succeed, basis might not be complete"
    return expanded_operator
end

# LinearAlgebra
# -------------

Base.:*(x::OperatorBasis, y::Number) = LocalOp(x) * y
Base.:*(x::Number, y::OperatorBasis) = x * LocalOp(y)
