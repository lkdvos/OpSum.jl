struct SiteOp{T, A, S}
    op::LocalOp{T, A}
    sites::Vector{S}
end

@doc """
    GlobalOp{T,A,S} <: SymbolicAlgebra{T}

Symbolic operator algebra for representing operators acting on (a subset of) a global Hilbert space.
The type perameter `T` is the scalar type, `A` denotes the algebra type, and `S` is the type of the space.
""" GlobalOp

@sumtype GlobalOp{T, A, S}(
    SiteOp{T, A, S},
    #   Scaled{T,GlobalOp{T,A,S}},
    Sum{T, GlobalOp{T, A, S}},
    Prod{GlobalOp{T, A, S}},
    Pow{GlobalOp{T, A, S}},
    Fun{GlobalOp{T, A, S}},
) <: SymbolicAlgebra{T}

# Syntax: O[inds...] means O applied to inds
function Base.getindex(O::LocalOp, ind::S, inds::S...) where {S}
    indices = S[ind, inds...]
    T = scalartype(O)
    A = algebratype(O)
    return GlobalOp{T, A, S}(SiteOp{T, A, S}(O, indices))
end

# Properties
# ----------
algebratype(::Type{GlobalOp{T, A, S}}) where {T, A, S} = A

sitetype(t) = sitetype(typeof(t))
sitetype(::Type{GlobalOp{T, A, S}}) where {T, A, S} = S
sitetype(::Type{T}) where {T} = throw(MethodError(sitetype, (T,)))

# LinearAlgebra
# -------------
function VectorInterface.add(x::GlobalOp, y::GlobalOp, α::Number, β::Number)
    (A = algebratype(x)) == algebratype(y) || throw(ArgumentError("incompatible algebra types."))
    (S = sitetype(x)) == sitetype(y) || throw(ArgumentError("incompatible site types"))
    T = VectorInterface.promote_add(x, y, α, β)
    z = GlobalOp{T, A, S}(Sum{T, GlobalOp{T, A, S}}())
    add!(z, x, β)
    add!(z, y, α)
    return z
end

function VectorInterface.add!(x::GlobalOp, y::GlobalOp, α::Number, β::Number)
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

function VectorInterface.add!!(x::GlobalOp, y::GlobalOp, α::Number, β::Number)
    if variant(x) isa Sum
        T = VectorInterface.promote_add(x, y, α, β)
        T === scalartype(x) && return add!(x, y, α, β)
    end
    return add(x, y, α, β)
end

function VectorInterface.scale(x::GlobalOp, α::Number)
    A = algebratype(x)
    S = sitetype(x)
    T = VectorInterface.promote_scale(x, α)
    z = GlobalOp{T, A, S}(Sum{T, GlobalOp{T, A, S}}())
    A = algebratype(x)
    z = GlobalOp{T, A, S}(Sum{T, GlobalOp{T, A, S}}())
    return scale!(z, x, α)
end

function VectorInterface.scale!(x::GlobalOp, α::Number)
    xvar = variant(x)
    @assert xvar isa Sum
    map!(Base.Fix2(scale, α), xvar.terms, xvar.terms)
    return x
end

function VectorInterface.scale!(y::GlobalOp, x::GlobalOp, α::Number)
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

LinearAlgebra.norm(x::GlobalOp) = sqrt(abs(inner(x, x)))
VectorInterface.inner(x::GlobalOp, y::GlobalOp) = inner(variant(x), variant(y))

function VectorInterface.inner(x::SiteOp, y::SiteOp)
    if x.sites == y.sites
        return inner(x.op, y.op)
    elseif isdisjoint(x.sites, y.sites)
        return inner(x.op, one(x.op)) * inner(one(y.op), y.op)
    else
        @assert variant(x.op) isa Kron && length(x.sites) == length(variant(x).ops) "TBA"
        @assert variant(y.op) isa Kron && length(y.sites) == length(variant(y).ops) "TBA"
        return prod(union(x.sites, y.sites)) do s
            i = findfirst(==(s), x.sites)
            j = findfirst(==(s), y.sites)
            if isnothing(i) && isnothing(j)
                return inner(one(x), one(y))
            elseif isnothing(i)
                O = variant(y.op).ops[j]
                return inner(one(O), O)
            elseif isnothing(j)
                O = variant(x.op).ops[i]
                return inner(O, one(O))
            else
                O₁ = variant(x.op).ops[i]
                O₂ = variant(y.op).ops[j]
                return inner(O₁, O₂)
            end
        end
    end
end

# Incorporating lattice information
# ---------------------------------

function Trie(vertices, ex::GlobalOp)
    A = algebratype(ex)
    V = scalartype(ex)
    root = Trie{A, V}()

    coefficients, opstrings = operatorstrings(vertices, ex)

    for (c, op) in zip(coefficients, opstrings)
        trie = root
        for o in op
            child = get!(trie.children, o) do
                Trie{A, V}()
            end
            trie = child
        end
        @assert isnothing(trie.value) "Duplicate values?"
        trie.value = c
    end

    return sortkeys!(root)
end
