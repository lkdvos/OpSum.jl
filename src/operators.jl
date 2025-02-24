"verify if this token represents the starting state"
isbegin(O::Enum) = O == first(instances(typeof(O)))
"verify if this token represents the ending state"
isend(O::Enum) = O == last(instances(typeof(O)))

function scalartype end
scalartype(x) = scalartype(typeof(x))
scalartype(::Type{T}) where {T} = throw(MethodError(scalartype, Tuple{T}))

function coefficient end
function coefficient! end

function operator_type end

module PauliOperators

using LinearAlgebra: norm

export PauliBasis, PauliOperator

import Base: +, *, -, /, one, zero
import ..OpSum: operator_type, scalartype, coefficient, coefficient!

# TODO: @algebragenerators
@enum PauliBasis B I X Y Z E
one(::PauliBasis) = I
one(::Type{PauliBasis}) = I

Base.to_index(x::PauliBasis) = Int(x) + 1

# TODO: make tuple/svector
struct PauliOperator{T<:Number}
    components::Vector{T}
end
function PauliOperator(vec::Vector{T}) where {T}
    length(vec) == length(instances(PauliBasis)) ||
        throw(ArgumentError("vec does not have the correct length"))
    return PauliOperator{T}(vec)
end

function PauliOperator{T}(x::PauliBasis) where {T<:Number}
    vec = zeros(T, length(instances(typeof(x))))
    vec[x] = 1
    return PauliOperator{T}(vec)
end
PauliOperator(x::PauliBasis) = PauliOperator{Bool}(x)
PauliOperator(x::AbstractVector) = PauliOperator(collect(x))

scalartype(::Type{PauliOperator{T}}) where {T} = T
operator_type(::Type{PauliBasis}, ::Type{T}=Bool) where {T} = PauliOperator{T}

# scalar multiplication/division
(λ::Number * O::PauliOperator) = PauliOperator(λ * O.components)
(O::PauliOperator * λ::Number) = λ * O

# addition/subtraction
(O1::PauliOperator + O2::PauliOperator) = PauliOperator(O1.components + O2.components)
(O1::PauliOperator - O2::PauliOperator) = PauliOperator(O1.components + O2.components)

one(x::PauliOperator) = one(typeof(x))
one(::Type{PauliOperator{T}}) where {T} = PauliOperator{T}(one(PauliBasis))
zero(x::PauliOperator) = zero(typeof(x))
zero(::Type{PauliOperator{T}}) where {T} = PauliOperator{T}(I) * false # TODO

coefficient!(x::PauliOperator) = (n = norm(x.components); x.components ./= n; n)
coefficient(x::PauliOperator) = norm(x.components)

function Base.show(io::IO, x::PauliOperator)
    iszero(x) && print(io, zero(scalartype(x)), one(PauliBasis))
    isone(x) && print(io, one(PauliBasis))
    inds = findall(!≈(0), x.components)
    if isempty(inds)
        print(io, zero(scalartype(x)))
    else
        join(
            io,
            [
                (isone(x.components[i]) ? "" : string(x.components[i])) *
                string(PauliBasis(i - 1)) for i in inds
            ],
            " + ",
        )
    end
    return nothing
end

end
