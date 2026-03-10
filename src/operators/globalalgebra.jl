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

# Instantiation
# -------------
function instantiate(O::GlobalOp{T, A, S}, sites) where {T, A, S}
    o = variant(O)
    if o isa SiteOp
        @assert issorted(o.sites) && allunique(o.sites) "Sites must be sorted and unique."

        # Identity operator (no site indices): embed as identity on all sites
        if isempty(o.sites)
            return mapfoldl(kron, eachindex(sites)) do i
                instantiate(one(o.op), sites[i])
            end
        end

        # Operator covers exactly the given sites: instantiate directly
        if length(o.sites) == length(sites)
            return instantiate(o.op, map(Base.Fix1(getindex, sites), o.sites))
        end

        # Partial embedding: determine per-site local factors
        op_var = variant(o.op)
        local_factors = if op_var isa Kron
            @assert length(op_var.factors) == length(o.sites)
            op_var.factors
        else
            @assert length(o.sites) == 1 "Non-Kron LocalOp must act on exactly one site"
            [o.op]
        end

        id_op = one(first(local_factors))
        return mapfoldl(kron, eachindex(sites)) do i
            j = findfirst(==(i), o.sites)
            isnothing(j) ? instantiate(id_op, sites[i]) : instantiate(local_factors[j], sites[i])
        end

    elseif o isa Sum
        return sum(pairs(o.terms)) do (k, v)
            return v * instantiate(k, sites)
        end
    elseif o isa Prod
        return prod(o.factors) do f
            return instantiate(f, sites)
        end
    elseif o isa Pow
        return instantiate(o.op, sites)^o.exponent
    elseif o isa Fun
        return o.f(map(x -> instantiate(x, sites), o.args)...)
    else
        error()
    end
end

# LinearAlgebra
# -------------

Base.zero(x::GlobalOp) = zero(typeof(x))
Base.zero(::Type{GlobalOp{T, A, S}}) where {T, A, S} = GlobalOp{T, A, S}(Sum{T, GlobalOp{T, A, S}}())
Base.one(::GlobalOp{T, A, S}) where {T, A, S} = GlobalOp{T, A, S}(SiteOp(LocalOp{T, A}(one(T)), S[]))

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
        @assert variant(x.op) isa Kron && length(x.sites) == length(variant(x.op).factors) || length(x.sites) == 1 "TBA: $x $y"
        @assert variant(y.op) isa Kron && length(y.sites) == length(variant(y.op).factors) || length(y.sites) == 1 "TBA: $x $y"
        return prod(union(x.sites, y.sites)) do s
            i = findfirst(==(s), x.sites)
            j = findfirst(==(s), y.sites)
            if isnothing(i) && isnothing(j)
                return inner(one(x), one(y))
            elseif isnothing(i)
                O = variant(y.op) isa Kron ? variant(y.op).factors[j] : y.op
                return inner(one(O), O)
            elseif isnothing(j)
                O = variant(x.op) isa Kron ? variant(x.op).factors[i] : x.op
                return inner(O, one(O))
            else
                O₁ = variant(x.op) isa Kron ? variant(x.op).factors[i] : x.op
                O₂ = variant(y.op) isa Kron ? variant(y.op).factors[j] : y.op
                return inner(O₁, O₂)
            end
        end
    end
end

# Operator products
# -----------------
function Base.:*(x::GlobalOp{T, A, S}, y::GlobalOp{T, A, S}) where {T, A, S}
    o1 = variant(x)
    o2 = variant(y)
    return if o1 isa SiteOp
        if o2 isa SiteOp
            return GlobalOp{T, A, S}(o1 * o2)
        elseif o2 isa Sum
            return sum(pairs(o2.terms)) do (k, v)
                return v * (x * k)
            end
        else
            error()
        end
    elseif o1 isa Sum
        return sum(pairs(o1.terms)) do (k, v)
            return v * (k * y)
        end
    else
        error()
    end
end

function Base.:*(x::SiteOp, y::SiteOp)
    if maximum(x.sites) < minimum(y.sites)
        sites = vcat(x.sites, y.sites)
        op = kron(x.op, y.op)
        I = (!isone).(op.factors)
        if !any(I)
            I[1] = true
        end
        keepat!(op.factors, I)
        keepat!(sites, I)
        return SiteOp(op, sites)
    elseif maximum(y.sites) < minimum(x.sites)
        # assume commutative if sites disjoint for now
        return y * x
    elseif x.sites == y.sites
        return SiteOp(x.op * y.op, x.sites)
    else
        error("TBA")
    end
end
function Base.:*(x::SiteOp{T, A, S}, y::Sum{T, GlobalOp{T, A, S}}) where {T, A, S}
    return sum(pairs(y.terms)) do (k, v)
        return v * (x * variant(k))
    end
end
function Base.:*(x::Sum{T, GlobalOp{T, A, S}}, y::SiteOp{T, A, S}) where {T, A, S}
    return sum(pairs(x.terms)) do (k, v)
        return v * (variant(k) * y)
    end
end


# Simplify
# --------
function simplify(x::GlobalOp{T, A, S}) where {T, A, S}
    o = variant(x)
    if o isa SiteOp
        op′ = simplify(o.op)
        iszero(op′) && return zero(x)
        return GlobalOp{T, A, S}(SiteOp{T, A, S}(op′, o.sites))
    elseif o isa Sum
        result = GlobalOp{T, A, S}(Sum{T, GlobalOp{T, A, S}}())
        result_terms = variant(result).terms
        for (k, v) in pairs(o.terms)
            iszero(v) && continue
            k′ = simplify(k)
            setwith!(+, result_terms, k′, v)
        end
        filter!(!iszero, result_terms)
        isempty(result_terms) && return zero(x)
        if length(result_terms) == 1
            k, v = only(pairs(result_terms))
            isone(v) && return k
        end
        return result
    elseif o isa Prod
        factors = GlobalOp{T, A, S}[]
        for f in o.factors
            push!(factors, simplify(f))
        end
        length(factors) == 1 && return only(factors)
        return GlobalOp{T, A, S}(Prod{GlobalOp{T, A, S}}(factors))
    elseif o isa Pow
        base = simplify(o.base)
        o.exponent == 1 && return base
        return GlobalOp{T, A, S}(Pow{GlobalOp{T, A, S}}(base, o.exponent))
    elseif o isa Fun
        args = map(simplify, o.args)
        return GlobalOp{T, A, S}(Fun{GlobalOp{T, A, S}}(o.f, args))
    else
        return x
    end
end

# Show
# ----
function Base.show(io::IO, operator::GlobalOp)
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

function Base.show_unquoted(io::IO, operator::GlobalOp, ::Int, precedence::Int)
    Base.show_unquoted(io, variant(operator), 0, precedence)
    return nothing
end

function Base.show(io::IO, operator::SiteOp{T, A, S}) where {T, A, S}
    print(io, "(")
    ioc = IOContext(io, :typeinfo => typeof(operator.op))
    print(ioc, operator.op)
    print(io, ")[")
    join(io, operator.sites, ", ")
    print(io, "]")
    return nothing
end

# Incorporating lattice information
# ---------------------------------

function Trie(vertices, ex::GlobalOp{T, A, S}) where {T, A, S}
    root = Trie{A, T}()

    coefficients, opstrings = operatorstrings(vertices, ex)

    for (c, op) in zip(coefficients, opstrings)
        trie = root
        for o in op
            child = get!(trie.children, o) do
                Trie{A, T}()
            end
            trie = child
        end
        @assert isnothing(trie.value) "Duplicate values?"
        trie.value = c
    end

    return sortkeys!(root)
end

function GraphNode(vertices::AbstractVector{Int}, ex::GlobalOp)
    @assert vertices == 1:length(vertices)
    A = algebratype(ex)
    T = scalartype(ex)

    root = GraphNode{A, T}()
    for (c, op) in zip(operatorstrings(vertices, ex)...)
        node = root
        for site in 1:length(vertices)
            child = typeof(root)(site == 1 ? c : nothing)
            push!(node.children, op[site] => [child])
            push!(child.parents, op[site] => node)
            node = child
        end
    end

    return root
end

function _emit_leaf!(node::Trie{A, T}, site_factors, coeff::T) where {A, T}
    for op in site_factors
        node = get!(() -> Trie{A, T}(), node.children, op)
    end
    node.value = isnothing(node.value) ? coeff : node.value + coeff
    return nothing
end

function build_trie!(
        trie::Trie{A, T}, vertices, ex::GlobalOp{T, A, S}, coeff::T
    ) where {T, A, S}
    iszero(coeff) && return trie
    o = variant(ex)

    if o isa Sum
        for (k, v) in pairs(o.terms)
            build_trie!(trie, vertices, k, coeff * v)
        end

    elseif o isa SiteOp
        @assert issorted(o.sites) && allunique(o.sites)

        if length(o.sites) == 1
            local_coeffs, local_ops = operatorstrings(o.op)
            site_pos = only(o.sites)
            site_factors = fill!(similar(vertices, A), one(A))
            for (lc, lop) in zip(local_coeffs, local_ops)
                iszero(lc) && continue
                site_factors[site_pos] = lop
                _emit_leaf!(trie, site_factors, coeff * lc)
            end

        elseif length(o.sites) == length(vertices)
            @assert variant(o.op) isa Kron
            local_coeffs, local_ops = operatorstrings(o.op)
            for (lc, lop) in zip(local_coeffs, local_ops)
                iszero(lc) && continue
                _emit_leaf!(trie, lop, coeff * lc)
            end

        else
            @assert variant(o.op) isa Kron && length(o.sites) == length(o.op.factors)
            ops = mapfoldl(kron, eachindex(vertices)) do i
                j = findfirst(==(i), o.sites)
                isnothing(j) ? one(o.op) : o.op.factors[j]
            end
            build_trie!(trie, vertices, GlobalOp(SiteOp(ops, collect(eachindex(vertices)))), coeff)
        end

    else
        error("build_trie!: unsupported GlobalOp variant $(typeof(o))")
    end
    return trie
end

"""
    opsum(f, itr)
    opsum(itr)

Like `sum`, but accumulates `GlobalOp` elements using `add!!` to enable in-place mutation
of the running total when the scalar type permits it.
"""
function opsum(f, itr)
    x, rest = Iterators.peel(itr)
    y = f(x)
    result = add(zero(y), y)  # first: allocate a fresh Sum
    for x in rest
        result = add!!(result, f(x))
    end
    return result
end
opsum(itr) = opsum(identity, itr)

function operatorstrings(vertices, O::GlobalOp{T, A, S}) where {T, A, S}
    coefficients = T[]
    opstrings = typeof(similar(vertices, A))[]

    o = variant(O)
    if o isa SiteOp
        @assert issorted(o.sites) && allunique(o.sites)
        if length(o.sites) == 1
            coeff, opstring = operatorstrings(o.op)
            append!(coefficients, coeff)
            for opstr in opstring
                opstr′ = fill!(similar(vertices, A), one(A))
                opstr′[only(o.sites)] = opstr
                push!(opstrings, opstr′)
            end
        elseif length(vertices) == length(o.sites)
            @assert variant(o.op) isa Kron
            coeff, opstring = operatorstrings(o.op)
            append!(coefficients, coeff)
            append!(opstrings, opstring)
        else
            @assert variant(o.op) isa Kron && length(o.sites) == length(o.op.factors)
            ops = mapfoldl(kron, eachindex(vertices)) do i
                j = findfirst(==(i), o.sites)
                if isnothing(j)
                    return one(o.op)
                else
                    return o.op.factors[j]
                end
            end
            return operatorstrings(vertices, GlobalOp(SiteOp(ops, collect(eachindex(vertices)))))
        end
    elseif o isa Sum
        for (k, v) in pairs(o.terms)
            coefficients′, opstrings′ = operatorstrings(vertices, k)
            append!(opstrings, opstrings′)
            append!(coefficients, coefficients′ .* v)
        end
    else
        error("TBA")
    end
    return coefficients, opstrings
end
