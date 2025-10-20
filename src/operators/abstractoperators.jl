"""
    SymbolicAlgebra{T<:Number}

Abstract supertype for working with symbolic (operator) algebras over a scalar field denoted
with `T`.
"""
abstract type SymbolicAlgebra{T <: Number} end

VectorInterface.scalartype(::Type{<:SymbolicAlgebra{T}}) where {T} = T

algebratype(x) = algebratype(typeof(x))
algebratype(T::Type) = throw(MethodError(algebratype, (T,)))

# Building blocks
# ---------------
# these are helper structs to denote the different ways to compose symbolic operators

# struct Scaled{T <: Number, O <: SymbolicAlgebra{T}}
#     op::O
#     scalar::T
# end

struct Sum{T <: Number, O <: SymbolicAlgebra{T}}
    terms::Dictionary{O, T}
    Sum{O, T}() where {O, T} = new{O, T}(Dictionary{O, T}())
    Sum{O, T}(terms::Dictionary{O, T}) where {O, T} = new{O, T}(terms)
end

struct Prod{O <: SymbolicAlgebra}
    factors::Vector{O}
    Prod{O}() where {O} = new{O}(O[])
    Prod{O}(factors::Vector{O}) where {O} = new{O}(factors)
end

struct Pow{O <: SymbolicAlgebra}
    base::O
    exponent::Int # should this be a generic number?
    Pow{O}(base::O, exponent::Int) where {O} = new{O}(base, exponent)
end

struct Kron{O <: SymbolicAlgebra}
    factors::Vector{O}
    Kron{O}() where {O} = new{O}(O[])
    Kron{O}(factors::Vector{O}) where {O} = new{O}(factors)
end

struct Fun{O <: SymbolicAlgebra}
    f::Any
    args::Vector{O}
    Fun{O}(f) where {O} = new{O}(f, O[])
    Fun{O}(f, args::Vector{O}) where {O} = new{O}(f, args)
end

# Linear Algebra
# --------------
Base.:+(x::SymbolicAlgebra) = scale(x, One())
Base.:+(x::SymbolicAlgebra, y::SymbolicAlgebra) = add(x, y)
Base.:-(x::SymbolicAlgebra) = scale(x, -1)
Base.:-(x::SymbolicAlgebra, y::SymbolicAlgebra) = add(x, y, -1)
Base.:*(x::SymbolicAlgebra, y::Number) = scale(x, y)
Base.:*(x::Number, y::SymbolicAlgebra) = scale(y, x)
Base.:/(x::SymbolicAlgebra, y::Number) = scale(x, inv(y))
Base.:\(x::Number, y::SymbolicAlgebra) = scale(y, inv(x))

function Base.rtoldefault(
        x::Union{T, Type{T}}, y::Union{S, Type{S}}, atol::Real
    ) where {T <: SymbolicAlgebra, S <: SymbolicAlgebra}
    return Base.rtoldefault(scalartype(x), scalartype(y), atol)
end

function Base.isapprox(
        x::SymbolicAlgebra, y::SymbolicAlgebra;
        atol::Real = 0, rtol::Real = Base.rtoldefault(x, y, atol)
    )
    return norm(x - y) <= max(atol, rtol * max(norm(x), norm(y)))
end

VectorInterface.scalartype(::Type{<:Sum{T}}) where {T} = T
VectorInterface.scalartype(::Type{<:Prod{O}}) where {O} = scalartype(O)
VectorInterface.scalartype(::Type{<:Kron{O}}) where {O} = scalartype(O)

function VectorInterface.inner(x::Sum, y::Sum)
    T = VectorInterface.promote_inner(x, y)
    result = zero(T)
    for (O₁, λ₁) in pairs(x.terms), (O₂, λ₂) in pairs(y.terms)
        result += @show inner(O₁, O₂) * @show inner(λ₁, λ₂)
    end
    return result
end

function VectorInterface.inner(x::Sum{T, O}, y::O) where {T, O}
    Tres = VectorInterface.promote_inner(scalartype(x), scalartype(y))
    result = zero(Tres)
    for (O₁, λ₁) in x.terms
        result += inner(O₁, y) * conj(λ₁)
    end
    return result
end
function VectorInterface.inner(x::O, y::Sum{T, O}) where {T, O}
    Tres = VectorInterface.promote_inner(scalartype(x), scalartype(y))
    result = zero(Tres)
    for (O₂, λ₂) in y.terms
        result += inner(x, O₂) * λ₂
    end
    return result
end

function VectorInterface.inner(x::Sum{T, O}, y::T) where {T, O}
    Tres = VectorInterface.promote_inner(scalartype(x), scalartype(y))
    result = zero(Tres)
    for (O₁, λ₁) in x.terms
        result += inner(O₁, one(O₁)) * conj(λ₁)
    end
    return result
end
function VectorInterface.inner(x::T, y::Sum{T, O}) where {T, O}
    Tres = VectorInterface.promote_inner(scalartype(x), scalartype(y))
    result = zero(Tres)
    for (O₂, λ₂) in y.terms
        result += inner(one(O₂), O₂) * λ₂
    end
    return result
end

# Show
# ----
function show_scaled(io::IO, operator, scalar, precedence::Int)
    should_parenthesize = !isone(scalar) && (!isreal(scalar) || !isone(abs(scalar))) &&
        Base.operator_precedence(:*) ≤ precedence

    should_parenthesize && print(io, '(')

    compact = get(io, :compact, false)
    print_type = !(get(io, :typeinfo, Any) <: typeof(operator))
    if print_type
        print(io, typeof(operator))
        if compact
            print(io, '(')
        else
            println(io, ':')
            print(io, " ")
        end

        io = IOContext(io, :typeinfo => typeof(operator))
    end

    if isone(scalar)
        show(io, operator)
    elseif isreal(scalar) && isone(abs(scalar))
        print(io, '-')
        show(io, operator)
    else
        Base.show_unquoted(io, scalar, 0, Base.operator_precedence(:*))
        print(io, " * ")
        Base.show_unquoted(io, operator, 0, Base.operator_precedence(:*))
    end

    print_type && compact && print(io, ')')
    should_parenthesize && print(io, ')')

    return nothing
end

function Base.show(io::IO, operator::Sum)
    precedence = Base.operator_precedence(:+)

    for (i, (op, scalar)) in enumerate(pairs(operator.terms))
        if i == 1
            show_scaled(io, op, scalar, precedence)
            continue
        end

        if isreal(scalar) && real(scalar) < 0
            print(io, " - ")
            scalar = abs(scalar)
        else
            print(io, " + ")
        end

        show_scaled(io, op, scalar, precedence)
    end

    return nothing
end

function Base.show_unquoted(io::IO, operator::Sum, ::Int, precedence::Int)
    if length(operator.terms) == 1
        op, scalar = only(pairs(operator.terms))
        show_scaled(io, op, scalar, precedence)
        return nothing
    end

    if Base.operator_precedence(:+) ≤ precedence
        print(io, "(")
        show(io, operator)
        print(io, ")")
    else
        show(io, operator)
    end

    return nothing
end
