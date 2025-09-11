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
