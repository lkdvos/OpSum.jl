module FermionOperators

export I, CC, CCdag, CdagC, CdagCdag, N

using OpSum: OperatorBasis, LocalOp
import OpSum: namemap, instantiate, CommutationType, Anticommuting
using VectorInterface

# local basis consists of: I, C†, C, N
primitive type FermionOperator <: OperatorBasis 8 end
function FermionOperator(x::Integer)
    0 <= x <= 3 || throw(DomainError(x, "invalid ID"))
    return Core.Intrinsics.bitcast(FermionOperator, convert(UInt8, x))
end

Base.Integer(x::FermionOperator) = Core.Intrinsics.bitcast(UInt8, x)

Base.isreal(::Type{FermionOperator}) = true

Base.isless(x::FermionOperator, y::FermionOperator) = isless(Integer(x), Integer(y))

VectorInterface.scalartype(::Type{FermionOperator}) = Bool
VectorInterface.inner(x::FermionOperator, y::FermionOperator) = x == y
namemap(::Type{FermionOperator}) = Dict{UInt8, Symbol}(0x00 => :I, 0x01 => :Cdag, 0x02 => :C, 0x03 => :N)

function instantiate(x::FermionOperator, ::Type{T}, axs) where {T}
    destination = fill!(similar(Array{T}, (axs..., axs...)), zero(T))
    size(destination) == (2, 2) || throw(DimensionMismatch())

    destination[Integer(x)] = 1
    x == FermionOperator(0x00) && (destination[4] = 1)
    return destination
end

Base.one(::Type{FermionOperator}) = FermionOperator(0x00)
CommutationType(::Type{FermionOperator}) = Anticommuting()

const I = LocalOp(FermionOperator(0x00))
const CdagCdag = LocalOp(FermionOperator(0x01)) ⊗ LocalOp(FermionOperator(0x01))
const CdagC = LocalOp(FermionOperator(0x01)) ⊗ LocalOp(FermionOperator(0x02))
const CCdag = -LocalOp(FermionOperator(0x02)) ⊗ LocalOp(FermionOperator(0x01))
const CC = -LocalOp(FermionOperator(0x02)) ⊗ LocalOp(FermionOperator(0x02))
const N = LocalOp(FermionOperator(0x03))

end
