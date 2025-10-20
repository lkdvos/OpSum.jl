module PauliOperators

export I, X, Y, Z

using OpSum: OperatorBasis, LocalOp
import OpSum: namemap, instantiate
using VectorInterface

primitive type PauliOperator <: OperatorBasis 8 end
function PauliOperator(x::Integer)
    0 <= x <= 3 || throw(DomainError(x, "invalid ID"))
    return Core.Intrinsics.bitcast(PauliOperator, convert(UInt8, x))
end

Base.Integer(x::PauliOperator) = Core.Intrinsics.bitcast(UInt8, x)

Base.isreal(::Type{PauliOperator}) = false

Base.isless(x::PauliOperator, y::PauliOperator) = isless(Integer(x), Integer(y))

VectorInterface.scalartype(::Type{PauliOperator}) = Complex{Bool}
VectorInterface.inner(x::PauliOperator, y::PauliOperator) = x == y
namemap(::Type{PauliOperator}) = Dict{UInt8, Symbol}(0x00 => :I, 0x01 => :X, 0x02 => :Y, 0x03 => :Z)

function instantiate(x::PauliOperator, ::Type{T}, axs) where {T}
    destination = fill!(similar(Array{T}, (axs..., axs...)), zero(T))
    # destination = zeros(T, (axs..., axs...))
    size(destination) == (2, 2) || throw(ArgumentError("only 2x2 pauli supported for now ($axs)"))

    if x == PauliOperator(0x00)
        destination[1, 1] = destination[2, 2] = 1
    elseif x == PauliOperator(0x01)
        destination[2, 1] = destination[1, 2] = 1
    elseif x == PauliOperator(0x02)
        destination[2, 1] = -(destination[1, 2] = im)
    elseif x == PauliOperator(0x03)
        destination[2, 2] = -(destination[1, 1] = 1)
    else
        error()
    end
    return destination
end

Base.one(::Type{PauliOperator}) = PauliOperator(0x00)

const I = LocalOp(PauliOperator(0x00))
const X = LocalOp(PauliOperator(0x01))
const Y = LocalOp(PauliOperator(0x02))
const Z = LocalOp(PauliOperator(0x03))

end
