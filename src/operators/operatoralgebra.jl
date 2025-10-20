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
        return mapreduce(kron, o.factors, sites) do f, s
            return instantiate(f, s)
        end
    else
        error("TBA")
    end
end

# LinearAlgebra
# -------------

Base.one(x::LocalOp) = one(typeof(x))
Base.one(::Type{LocalOp{T, A}}) where {T, A} = LocalOp{T, A}(one(T))
Base.zero(x::LocalOp) = zero(typeof(x))
Base.zero(::Type{LocalOp{T, A}}) where {T, A} = LocalOp{T, A}(zero(T))

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

VectorInterface.inner(x::LocalOp, y::LocalOp) = inner(variant(x), variant(y))
LinearAlgebra.norm(x::LocalOp) = sqrt(abs(inner(x, x)))


# Algebra
# -------
function LinearAlgebra.kron(x::L, y::L) where {L <: LocalOp}
    result = Kron{L}(L[])
    variant(x) isa Kron ? append!(result.factors, x.factors) : push!(result.factors, x)
    variant(y) isa Kron ? append!(result.factors, y.factors) : push!(result.factors, y)
    return LocalOp(result)
end


# Incorporating lattice information
# ---------------------------------
function operatorstrings(x::LocalOp{T, A}) where {T, A}
    if variant(x) isa T
        return [variant(x)], [one(A)]
    elseif variant(x) isa A
        return [one(T)], [variant(x)]
    elseif variant(x) isa Sum
        coeffs = T[]
        opstrings = Vector{A}[]
        for (k, v) in pairs(variant(x).terms)
            coeff, opstring = operatorstrings(k)
            append!(coeffs, v .* coeff)
            append!(opstrings, opstring)
        end
        return coeffs, opstrings
    elseif variant(x) isa Kron
        λ = one(T)
        opstring = map(variant(x).factors) do f
            fvar = variant(f)
            if fvar isa A
                return fvar
            elseif fvar isa T
                λ *= fvar
                return one(A)
            else
                error("$(typeof(f))")
            end
        end
        return [λ], Vector{A}[opstring]
    else
        error("unsupported $(typeof(variant(x)))")
    end
end

# Show
# ----
function Base.show(io::IO, operator::LocalOp)
    compact = get(io, :compact, false)
    print_type = !(get(io, :typeinfo, Any) <: typeof(operator))
    if print_type
        print(io, typeof(operator))
        if compact
            print(io, "(")
        else
            println(io, ":")
            print(io, " ")
        end
        io = IOContext(io, :typeinfo => typeof(operator))
    end

    show(io, variant(operator))

    if print_type && compact
        print(io, ")")
    end
    return nothing
end

function Base.show_unquoted(io::IO, operator::LocalOp, ::Int, precedence::Int)
    Base.show_unquoted(io, variant(operator), 0, precedence)
    return nothing
end
