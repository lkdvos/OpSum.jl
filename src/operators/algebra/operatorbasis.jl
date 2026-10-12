# Declarations the `IrrepTensorOperators` submodule subtypes / extends.

"""
    abstract type OperatorBasis

Supertype for on-site operator alphabets; the one concrete alphabet is [`IrrepOperator`](@ref).
"""
abstract type OperatorBasis end

"""
    instantiate(op, V)

Materialize a symbolic operator as a `TensorMap` on the physical space `V`.
"""
function instantiate end
