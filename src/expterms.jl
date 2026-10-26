# Exponentially decaying interactions `Σ_{i<j} λ^{j-i-1} A_i S_{i+1}…S_{j-1} B_j`, `|λ| < 1`, as
# `channels` the sweep lowers to a single bond index carrying `λ` on its diagonal. A channel is one
# fully specified representative term plus the site where its exit block starts. With lattice period
# `P` it stands for every translate by multiples of `P` that fits, exit block pushed right; a
# translate weighs `coeff · λ^{string sites}`, the string sitting only on the stretched gap.

using TensorKit: Sector, unit

export expterm

# --- the descriptor -------------------------------------------------------------------------------

# One decaying interaction: the representative `term`, the `exitsite` where its exit block starts
# (sites `< exitsite` are the entry block; both non-empty), `decay` λ, and the `string` carried by the
# stretched gap as a canonical `(letter, coefficient)` list (`nothing` = identity).
struct ExpKey{I <: Sector}
    term::Term{I}
    exitsite::Int
    decay::ComplexF64
    string::Vector{Tuple{Union{Nothing, Letter{I}}, ComplexF64}}
end

function Base.:(==)(x::ExpKey{I}, y::ExpKey{I}) where {I}
    return x.term == y.term && x.exitsite == y.exitsite && x.decay == y.decay &&
        x.string == y.string
end
function Base.hash(x::ExpKey, h::UInt)
    return hash(x.string, hash(x.decay, hash(x.exitsite, hash(x.term, hash(:ExpKey, h)))))
end

function Base.show(io::IO, k::ExpKey)
    print(io, "ExpKey(", k.term, ", exitsite=", k.exitsite, ", decay=", k.decay)
    (length(k.string) == 1 && only(k.string)[1] === nothing && isone(only(k.string)[2])) ||
        print(io, ", string=", k.string)
    return print(io, ")")
end

channelspan(k::ExpKey) = Int(maximum(k.term.sites) - minimum(k.term.sites))

function translate(k::ExpKey{I}, Δ) where {I}
    iszero(Δ) && return k
    term = Term{I}(k.term.sites .+ Int(Δ), k.term.factors, k.term.coeff)
    return ExpKey{I}(term, k.exitsite + Int(Δ), k.decay, k.string)
end

# --- the containers -------------------------------------------------------------------------------

"""
    ExpSum{I<:Sector}

A sum of exponentially decaying interactions, built by [`expterm`](@ref) and added to an
[`OperatorSum`](@ref). Each distinct decay `λ` costs its own bond index.
"""
struct ExpSum{I <: Sector}
    channels::Dictionary{ExpKey{I}, ComplexF64}
end
ExpSum{I}() where {I} = ExpSum{I}(Dictionary{ExpKey{I}, ComplexF64}())

Base.length(es::ExpSum) = length(es.channels)
Base.isempty(es::ExpSum) = isempty(es.channels)

function Base.show(io::IO, es::ExpSum)
    print(io, "ExpSum(")
    join(io, ("$v * $k" for (k, v) in pairs(es.channels)), " + ")
    return print(io, ")")
end

function Base.:+(a::ExpSum{I}, b::ExpSum{I}) where {I}
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for x in (a, b), (k, v) in pairs(x.channels)
        setwith!(+, d, k, ComplexF64(v))
    end
    filter!(!iszero, d)
    return ExpSum{I}(d)
end

function VectorInterface.scale(a::ExpSum{I}, α::Number) where {I}
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for (k, v) in pairs(a.channels)
        iszero(α * v) || insert!(d, k, ComplexF64(α * v))
    end
    return ExpSum{I}(d)
end
Base.:*(α::Number, a::ExpSum) = scale(a, α)
Base.:*(a::ExpSum, α::Number) = scale(a, α)
Base.:/(a::ExpSum, α::Number) = scale(a, inv(α))
Base.:-(a::ExpSum) = scale(a, -1)
Base.:-(a::ExpSum, b::ExpSum) = a + (-b)

# --- the constructor ------------------------------------------------------------------------------

# Canonical `(letter, coefficient)` list; letters must be neutral, else the loop is not a self-loop.
function _string_terms(::Type{I}, op) where {I <: Sector}
    A = Letter{I}
    terms = if op === nothing
        Tuple{Union{Nothing, A}, ComplexF64}[(nothing, one(ComplexF64))]
    else
        Tuple{Union{Nothing, A}, ComplexF64}[
            (ispassthrough(letter) ? nothing : letter, ComplexF64(c)) for (letter, c) in pairs(op) if !iszero(c)
        ]
    end
    isempty(terms) && throw(ArgumentError("expterm: the string operator is zero"))
    for (letter, _) in terms
        (letter === nothing || letter.c == unit(I)) || throw(
            ArgumentError(
                "expterm: the string operator must be charge-neutral, got a letter of charge " *
                    "$(letter.c). A charged string makes the running bond charge drift along the " *
                    "string, so the channel is no longer a self-loop."
            )
        )
    end
    lt(a, b) = a[1] === nothing ? b[1] !== nothing : (b[1] !== nothing && isless(a[1], b[1]))
    return sort!(terms; lt)
end

"""
    expterm(t::Terms; decay, exitsite = maximum(sites), string = nothing) -> ExpSum

An exponentially decaying interaction generated from the representative term(s) `t`, stretching the
gap just before `exitsite` geometrically with ratio `decay` (`0 < |decay| < 1`).
`t` must be fully coupled on at least two sites; `exitsite` is one of its active sites, not the first.
`string` is a charge-neutral `SiteOperator` on every site of the stretched gap (default identity).
The translation period is `L` on an [`InfiniteChain`](@ref), one site on a finite chain.
"""
function expterm(
        t::Terms{I}; decay, exitsite = nothing, string = nothing
    ) where {I}
    λ = ComplexF64(decay)
    iszero(λ) && throw(
        ArgumentError(
            "expterm: decay = 0 leaves only the shortest translate; write that as an ordinary term"
        )
    )
    abs(λ) < 1 || throw(
        ArgumentError("expterm: need |decay| < 1 for a summable interaction, got |λ| = $(abs(λ))")
    )
    isempty(t.terms) && throw(ArgumentError("expterm: the representative has no terms"))
    isempty(t.pending) || throw(
        ArgumentError(
            "expterm: the representative holds an unresolved on-site product; write the on-site " *
                "operator as one operator in the site's basis instead"
        )
    )
    str = _string_terms(I, string)

    d = Dictionary{ExpKey{I}, ComplexF64}()
    for term in t.terms
        # the coefficient rides in the dictionary, so the stored representative is normalised to 1
        k = Term{I}(term.sites, term.factors, one(ComplexF64))
        v = term.coeff
        length(k.sites) >= 2 || throw(
            ArgumentError(
                "expterm: the representative needs at least two active sites (an entry and an exit " *
                    "block), got sites $(k.sites)"
            )
        )
        es = exitsite === nothing ? maximum(k.sites) : Int(exitsite)
        es in k.sites || throw(
            ArgumentError("expterm: exitsite = $es is not an active site of the representative $(k.sites)")
        )
        es > minimum(k.sites) || throw(
            ArgumentError(
                "expterm: exitsite = $es leaves an empty entry block; it must lie to the right of " *
                    "the representative's first active site"
            )
        )
        setwith!(+, d, ExpKey{I}(k, es, λ, str), ComplexF64(v))
    end
    filter!(!iszero, d)
    return ExpSum{I}(d)
end

# --- explicit expansion (the oracle) --------------------------------------------------------------

# Every translate of channel `k` inside `1:N` (period `P`) as ordinary terms: exit block pushed right in steps of `P`.
function _expand_channel!(
        d::Dictionary{Support{I}, ComplexF64}, k::ExpKey{I}, coeff, P::Int, N::Int
    ) where {I}
    sites = k.term.sites
    ops = [key.letter for key in k.term.factors]
    bonds = [key.bond for key in k.term.factors]
    ne = count(<(k.exitsite), sites)                     # size of the entry block
    entry, exit_ = sites[1:ne], sites[(ne + 1):end]
    entrybond = bonds[ne]                                # running charge along the string
    g0 = Int(k.exitsite - maximum(filter(<(k.exitsite), k.term.sites))) - 1  # string sites in the shortest translate
    a, zlast = Int(minimum(sites)), Int(maximum(sites))

    for m in cld(1 - a, P):fld(N - zlast, P)             # entry anchor translate
        Δ = m * P
        nstr = g0
        while true
            newexit = exit_ .+ (Δ + nstr - g0)
            Int(maximum(newexit)) <= N || break
            w = ComplexF64(coeff * k.decay^nstr)
            strsites = (Int(maximum(entry)) + Δ + 1):(Int(minimum(newexit)) - 1)
            @assert length(strsites) == nstr
            _emit_stretched!(
                d, entry .+ Δ, newexit, ops, bonds, ne, entrybond, strsites, k.string, w
            )
            nstr += P
        end
    end
    return d
end

# One stretched translate: a composite string distributes over its letters at every gap site.
function _emit_stretched!(
        d::Dictionary{Support{I}, ComplexF64}, entry, exit_, ops, bonds, ne, entrybond,
        strsites, string, w
    ) where {I}
    nstr = length(strsites)
    active = filter(p -> p[1] !== nothing, string)
    if isempty(active) || nstr == 0
        # all gap sites idle: one term, with the string's scalar part once per site
        scalar = sum(c for (letter, c) in string if letter === nothing; init = zero(ComplexF64))
        (nstr == 0 || !iszero(scalar)) || return d
        wtot = nstr == 0 ? w : w * scalar^nstr
        newsites = Int[entry..., exit_...]
        newops = Letter{I}[ops...]
        newbonds = eltype(bonds)[bonds...]
        _insert_stretched!(d, newsites, newops, newbonds, wtot)
        return d
    end
    for combo in Iterators.product(ntuple(_ -> string, nstr)...)
        wtot = w
        strs = Tuple{Int, Letter{I}}[]
        for (s, (letter, c)) in zip(strsites, combo)
            wtot *= c
            letter === nothing || push!(strs, (s, letter))
        end
        iszero(wtot) && continue
        newsites = Int[entry..., first.(strs)..., exit_...]
        newops = Letter{I}[ops[1:ne]..., last.(strs)..., ops[(ne + 1):end]...]
        newbonds = eltype(bonds)[
            bonds[1:ne]..., ntuple(_ -> entrybond, length(strs))..., bonds[(ne + 1):end]...,
        ]
        _insert_stretched!(d, newsites, newops, newbonds, wtot)
    end
    return d
end

function _insert_stretched!(
        d::Dictionary{Support{I}, ComplexF64}, newsites, newops, newbonds, w
    ) where {I}
    # multiplicity-free fusion: every vertex label is 1
    keys_ = Factor{I}[Factor{I}(o, b) for (o, b) in zip(newops, newbonds)]
    # keyed by support, so `setwith!` sums coincident terms
    setwith!(+, d, (newsites, keys_), ComplexF64(w))
    return d
end

# Every translate of every channel inside `1:N` as an ordinary bag; exponential in the gap (oracle).
function expand_channels(es::ExpSum{I}, P::Int, N::Int) where {I}
    d = Dictionary{Support{I}, ComplexF64}()
    for (k, v) in pairs(es.channels)
        _expand_channel!(d, k, v, P, N)
    end
    filter!(!iszero, d)
    return Terms{I}([Term{I}(sites, factors, v) for ((sites, factors), v) in pairs(d)])
end
