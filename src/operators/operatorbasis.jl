"""
    abstract type OperatorBasis

Abstract supertype for all operator basis elements (the on-site alphabet letters wrapped by
`LocalOp`).
"""
abstract type OperatorBasis end

# Instantiation
# -------------
"""
    instantiate(op, V)

Materialize a basis element / symbolic operator into its concrete form on the physical space `V`
(a TensorKit `TensorMap` for the ITO alphabet). Implemented per alphabet; see
`instantiate(::IrrepOperator, ::ElementarySpace)`.
"""
function instantiate end

# Scalar scaling: a bare letter times a scalar promotes to a `LocalOp`. Used by the per-bond sweep
# when weighting a letter by its reduced coefficient (`k.op * coeff`).
Base.:*(x::OperatorBasis, y::Number) = LocalOp(x) * y
Base.:*(x::Number, y::OperatorBasis) = x * LocalOp(y)
