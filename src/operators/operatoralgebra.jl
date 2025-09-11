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

function LocalOp(op::OperatorBasis)
    A = typeof(op)
    T = isreal(A) ? Float64 : ComplexF64
    return LocalOp{T, A}(op)
end
LocalOp{T}(op::OperatorBasis) where {T <: Number} = LocalOp{T, typeof(op)}(op)

LocalOp{T, A}(op::LocalOp{T, A}) where {T, A} = deepcopy(op)

# Properties
# ----------
algebratype(::Type{LocalOp{T, A}}) where {T, A} = A

# Instantiation
# -------------
function instantiate(O::LocalOp{T, A}, sites) where {T, A}
    o = variant(O)
    if o isa T
        return instantiate(o * one(A), sites)
    elseif o isa A
        return instantiate(o, T, sites)
    elseif o isa Sum
        return sum(pairs(o.terms)) do (k, v)
            return v * instantiate(k, sites)
        end
    elseif o isa Prod
        return prod(o.factors) do f
            return instantiate(f, sites)
        end
    elseif o isa Kron
        return mapreduce(kron, o.factors) do f
            return instantiate(f, sites)
        end
    else
        error("TBA")
    end
end

# LinearAlgebra
# -------------
function VectorInterface.add(x::LocalOp, y::LocalOp, α::Number, β::Number)
    @assert algebratype(x) == algebratype(y)
    A = algebratype(x)
    T = VectorInterface.promote_add(x, y, α, β)
    z = LocalOp{T, A}(Sum{T, LocalOp{T, A}}())
    add!(z, x, β)
    add!(z, y, α)
    return z
end

function VectorInterface.add!(x::LocalOp, y::LocalOp, α::Number, β::Number)
    xvar = variant(x)
    @assert xvar isa Sum

    scale!(x, β)

    yvar = variant(y)
    if yvar isa Sum
        for (o, λ) in pairs(yvar.terms)
            λα = λ * α
            iszero(λα) || setwith!(+, xvar.terms, o, λα)
        end
    else
        iszero(α) || setwith!(+, xvar.terms, y, α)
    end

    return x
end

function VectorInterface.add!!(x::LocalOp, y::LocalOp, α::Number, β::Number)
    if variant(x) isa Sum
        T = VectorInterface.promote_add(x, y, α, β)
        T === scalartype(x) && return add!(x, y, α, β)
    end
    return add(x, y, α, β)
end

function VectorInterface.scale(x::LocalOp, α::Number)
    T = VectorInterface.promote_scale(x, α)
    A = algebratype(x)
    z = LocalOp{T, A}(Sum{T, LocalOp{T, A}}())
    return scale!(z, x, α)
end

function VectorInterface.scale!(x::LocalOp, α::Number)
    xvar = variant(x)
    @assert xvar isa Sum
    map!(Base.Fix2(scale, α), xvar.terms, xvar.terms)
    return x
end

function VectorInterface.scale!(y::LocalOp, x::LocalOp, α::Number)
    yvar = variant(y)
    @assert yvar isa Sum
    empty!(yvar.terms)

    xvar = variant(x)
    if xvar isa Sum
        for (o, λ) in pairs(xvar.terms)
            λα = λ * α
            iszero(λα) || insert!(yvar.terms, o, λα)
        end
    else
        iszero(α) || insert!(yvar.terms, x, α)
    end
    return y
end
