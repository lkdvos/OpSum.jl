@doc """
    LocalOp{T,A}

Symbolic operator algebra for representing local operators acting on a a number of sites.
The type parameter `T` denotes the scalar type of the operator, while `A` denotes the algebra.
""" LocalOp

@sumtype LocalOp{T, A}(
    T,
    A,
    #   Scaled{T,LocalOp{T,A}},
    Kron{LocalOp{T, A}},
    Sum{T, LocalOp{T, A}},
    Prod{LocalOp{T, A}},
    Pow{LocalOp{T, A}},
    Fun{LocalOp{T, A}},
) <: SymbolicAlgebra{T}

# Convenience constructors
# ------------------------
LocalOp(op::T) where {T <: Number} = LocalOp{T, Symbol}(op)
LocalOp(op::A) where {A <: Union{Symbol, Enum}} = LocalOp{Float64, A}(op)
LocalOp{T}(op::A) where {T <: Number, A <: Union{Symbol, Enum}} = LocalOp{T, A}(op)
LocalOp{T, A}(op::LocalOp{T, A}) where {T, A} = deepcopy(op)
