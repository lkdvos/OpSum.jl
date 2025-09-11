"""
    abstract type OperatorBasis

Abstract supertype for all operator basis elements.
"""
abstract type OperatorBasis end

function namemap end

function Base.show(io::IO, x::OperatorBasis)
    maps = namemap(typeof(x))
    return show(io, maps[x])
end


# Concrete operator bases
# -----------------------
module PauliOperators

    using OpSum: OperatorBasis
    import OpSum: namemap

    primitive type PauliOperator <: OperatorBasis 8 end
    function PauliOperator(x::Integer)
        0 <= x <= 3 || throw(DomainError(x, "invalid ID"))
        return Core.Intrinsics.bitcast(PauliOperator, convert(UInt8, x))
    end

    namemap(::Type{PauliOperator}) = Dict{UInt8, Symbol}(0x00 => :I, 0x01 => :X, 0x02 => :Y, 0x03 => :Z)
    const I = PauliOperator(0x00)
    const X = PauliOperator(0x01)
    const Y = PauliOperator(0x02)
    const Z = PauliOperator(0x03)

end
