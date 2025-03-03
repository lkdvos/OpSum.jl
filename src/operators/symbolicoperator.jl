"""
    SymbolicOperator{I,S,C<:Number}

This module defines the `SymbolicOperator` type, which is used to represent a symbolic operator algebra.
The elementary operators are defined through labels of type `I`, which can be any type.
These operators act on sites of type `S`, which can also be any type.
The coefficients of the operators are of type `C`, which is restricted to be a subtype of `Number`.

Finally, the different flavours of the types in this module ensure that these operators
function as a (symbolic) algebra over `C`, with scalar multiplication, addition, and multiplication.
"""
@data SymbolicOperator{I,S,C<:Number} begin
    # Single operator
    struct Operator
        flavour::I
        site::S
    end

    # Scalar multiplication
    struct ScaledOperator
        scalar::C
        operator::SymbolicOperator{I,S,C}
    end

    # Addition
    struct OperatorSum
        arguments::Dictionary{SymbolicOperator{I,S,C},C}
    end

    # Multiplication
    struct OperatorProduct
        arguments::Vector{SymbolicOperator{I,S,C}}
    end
end
