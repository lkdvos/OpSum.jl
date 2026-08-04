# The on-site alphabet extension point
# =====================================
# Two declarations that have to exist before the alphabet itself, because the `IrrepTensorOperators`
# submodule subtypes the one and adds methods to the other.

"""
    abstract type OperatorBasis

Supertype for on-site operator alphabets — the letters an [`OnsiteOp`](@ref) is a combination of.
The one concrete alphabet is [`IrrepOperator`](@ref), the irreducible tensor operators of a
`TensorKit` sector.
"""
abstract type OperatorBasis end

"""
    instantiate(op, V)

Materialize a symbolic operator into its concrete form on the physical space `V` (a TensorKit
`TensorMap`). Implemented per alphabet; see `instantiate(::IrrepOperator, ::ElementarySpace)`.
"""
function instantiate end
