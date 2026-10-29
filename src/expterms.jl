# Exponentially decaying interactions `Σ_{i<j} λ^{j-i-1} A_i S_{i+1}…S_{j-1} B_j`, `|λ| < 1`, as
# `channels` the sweep lowers to a single bond index carrying `λ` on its diagonal. A channel is one
# fully specified representative term plus the site where its exit block starts. With lattice period
# `P` it stands for every translate by multiples of `P` that fits, exit block pushed right; a
# translate weighs `coeff · λ^{string sites}`, the string sitting only on the stretched gap.

using TensorKit: Sector, unit

export expterm

# --- the descriptor -------------------------------------------------------------------------------

# One decaying interaction: the representative `support` (sites and factors; the coefficient rides in
# the `ExpSum`), the `exitsite` where its exit block starts (sites `< exitsite` are the entry block; both
# non-empty), `decay` λ, and the charge-neutral `string` operator carried by every site of the stretched
# gap (the identity, `one(SiteOperator{I})`, by default).
struct ExpKey{I <: Sector}
    support::Support{I}
    exitsite::Int
    decay::ComplexF64
    string::SiteOperator{I}
end

function Base.:(==)(x::ExpKey{I}, y::ExpKey{I}) where {I}
    return x.support == y.support && x.exitsite == y.exitsite && x.decay == y.decay &&
        x.string == y.string
end
function Base.hash(x::ExpKey, h::UInt)
    return hash(x.string, hash(x.decay, hash(x.exitsite, hash(x.support, hash(:ExpKey, h)))))
end

# Content order, so that the lowering (hence the bond order) does not depend on the order channels were written.
function Base.isless(x::ExpKey{I}, y::ExpKey{I}) where {I}
    return isless(_orderkey(x), _orderkey(y))
end
function _orderkey(k::ExpKey)
    sites, factors = k.support
    return (sites, k.exitsite, factors, reim(k.decay), k.string.letters, reim.(k.string.coeffs))
end

function Base.show(io::IO, k::ExpKey)
    sites, factors = k.support
    print(io, "ExpKey(sites=", sites, ", ops=", [f.letter for f in factors], ", exitsite=", k.exitsite, ", decay=", k.decay)
    isone(k.string) || print(io, ", string=", k.string)
    return print(io, ")")
end

channelspan(k::ExpKey) = Int(maximum(k.support[1]) - minimum(k.support[1]))

function translate(k::ExpKey{I}, Δ) where {I}
    iszero(Δ) && return k
    sites, factors = k.support
    return ExpKey{I}((sites .+ Int(Δ), factors), k.exitsite + Int(Δ), k.decay, k.string)
end

# --- the containers -------------------------------------------------------------------------------

"""
    ExpSum{I<:Sector}

A sum of exponentially decaying interactions, built by [`expterm`](@ref) and added to an
[`OperatorSum`](@ref). Each distinct decay `λ` costs its own bond index. `==` is exact and `≈` compares
the coefficients of equal descriptors; neither depends on the order the channels were added in.
"""
struct ExpSum{I <: Sector}
    channels::Dictionary{ExpKey{I}, ComplexF64}
end
ExpSum{I}() where {I} = ExpSum{I}(Dictionary{ExpKey{I}, ComplexF64}())

Base.length(es::ExpSum) = length(es.channels)
Base.isempty(es::ExpSum) = isempty(es.channels)
Base.copy(es::ExpSum{I}) where {I} = ExpSum{I}(copy(es.channels))

function Base.:(==)(a::ExpSum{I}, b::ExpSum{I}) where {I}
    length(a) == length(b) || return false
    return all(((k, v),) -> haskey(b.channels, k) && b.channels[k] == v, pairs(a.channels))
end
# order-independent: the dictionary iterates in insertion order
function Base.hash(es::ExpSum, h::UInt)
    acc = zero(UInt)
    for (k, v) in pairs(es.channels)
        acc ⊻= hash(v, hash(k))
    end
    return hash(acc, hash(length(es), hash(:ExpSum, h)))
end

function Base.show(io::IO, es::ExpSum)
    print(io, "ExpSum(")
    join(io, ("$v * $k" for (k, v) in pairs(es.channels)), " + ")
    return print(io, ")")
end

# Add `v` to the coefficient of `k` (inserting it if new); a coefficient that cancels is dropped.
function addchannel!(es::ExpSum{I}, k::ExpKey{I}, v::Number) where {I}
    c = get(es.channels, k, zero(ComplexF64)) + ComplexF64(v)
    if iszero(c)
        haskey(es.channels, k) && delete!(es.channels, k)
    else
        set!(es.channels, k, c)
    end
    return es
end
function addchannels!(es::ExpSum{I}, other::ExpSum{I}) where {I}
    for (k, v) in pairs(other.channels)
        addchannel!(es, k, v)
    end
    return es
end

function Base.:+(a::ExpSum{I}, b::ExpSum{I}) where {I}
    return addchannels!(addchannels!(ExpSum{I}(), a), b)
end

# close as sparse vectors over the descriptors: the same keys, coefficients close
Base.isapprox(a::ExpSum{I}, b::ExpSum{I}; kwargs...) where {I} =
    length(a) == length(b) &&
    all(((k, v),) -> haskey(b.channels, k) && isapprox(v, b.channels[k]; kwargs...), pairs(a.channels))

function VectorInterface.scale(a::ExpSum{I}, α::Number) where {I}
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for (k, v) in pairs(a.channels)
        iszero(α * v) || insert!(d, k, ComplexF64(α * v))
    end
    return ExpSum{I}(d)
end

# --- collecting -----------------------------------------------------------------------------------

# An `OperatorSum`'s collector carries an `ExpSum` in the bag's channel slot; a `Terms` has none.
function collect_terms!(out::TermBag{I, ExpSum{I}}, es::ExpSum{I}, ::Type{I}) where {I}
    addchannels!(out.channels, es)
    return out
end
collect_terms!(::TermBag{I, Nothing}, ::ExpSum{I}, ::Type{I}) where {I} = throw(
    ArgumentError(
        "a Terms has no decaying channels: add the ExpSum to an OperatorSum with `opsum` or `opsum!`"
    )
)
collect_terms!(out::TermBag{I}, ::ExpSum{J}, ::Type{I}) where {I, J} = wrongsector(out, J)
collect_any(::Nothing, es::ExpSum{I}) where {I} = collect_terms!(TermBag{I}(), es, I)

# --- the constructor ------------------------------------------------------------------------------

# The canonical string operator; its letters must be neutral, else the loop is not a self-loop.
function channel_string(::Type{I}, op) where {I <: Sector}
    op === nothing && return one(SiteOperator{I})
    str = convert(SiteOperator{I}, op)
    iszero(str) && throw(ArgumentError("expterm: the string operator is zero"))
    for letter in str.letters
        letter.c == unit(I) || throw(
            ArgumentError(
                "expterm: the string operator must be charge-neutral, got a letter of charge " *
                    "$(letter.c). A charged string makes the running bond charge drift along the " *
                    "string, so the channel is no longer a self-loop."
            )
        )
    end
    return str
end

"""
    expterm(t::Terms; decay, exitsite = maximum(sites), string = nothing) -> ExpSum

An exponentially decaying interaction generated from the representative term(s) `t`, stretching the
gap just before `exitsite` geometrically with ratio `decay` (`0 < |decay| < 1`).
`t` must be fully coupled on at least two sites; `exitsite` is one of its active sites, not the first.
`string` is a charge-neutral `SiteOperator` on every site of the stretched gap (default the identity).
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
    str = channel_string(I, string)

    d = Dictionary{ExpKey{I}, ComplexF64}()
    for term in t.terms
        # the coefficient rides in the dictionary, not in the key
        sites = term.sites
        v = term.coeff
        length(sites) >= 2 || throw(
            ArgumentError(
                "expterm: the representative needs at least two active sites (an entry and an exit " *
                    "block), got sites $sites"
            )
        )
        es = exitsite === nothing ? maximum(sites) : Int(exitsite)
        es in sites || throw(
            ArgumentError("expterm: exitsite = $es is not an active site of the representative $sites")
        )
        es > minimum(sites) || throw(
            ArgumentError(
                "expterm: exitsite = $es leaves an empty entry block; it must lie to the right of " *
                    "the representative's first active site"
            )
        )
        setwith!(+, d, ExpKey{I}(support(term), es, λ, str), ComplexF64(v))
    end
    filter!(!iszero, d)
    return ExpSum{I}(d)
end

# --- explicit expansion (the oracle) --------------------------------------------------------------

# Every translate of channel `k` inside `1:N` (period `P`) as ordinary terms: exit block pushed right in steps of `P`.
function _expand_channel!(
        d::Dictionary{Support{I}, ComplexF64}, k::ExpKey{I}, coeff, P::Int, N::Int
    ) where {I}
    sites, factors = k.support
    ops = [key.letter for key in factors]
    bonds = [key.bond for key in factors]
    ne = count(<(k.exitsite), sites)                     # size of the entry block
    entry, exit_ = sites[1:ne], sites[(ne + 1):end]
    entrybond = bonds[ne]                                # running charge along the string
    g0 = Int(k.exitsite - maximum(filter(<(k.exitsite), sites))) - 1  # string sites in the shortest translate
    strpairs = collect(pairs(k.string))                  # (letter => coefficient), identity included
    a, zlast = Int(minimum(sites)), Int(maximum(sites))

    for m in cld(1 - a, P):fld(N - zlast, P)             # entry anchor translate
        Δ = m * P
        nstr = g0
        while true
            newexit = exit_ .+ (Δ + nstr - g0)
            Int(maximum(newexit)) <= N || break
            w = ComplexF64(coeff * k.decay^nstr)
            strsites = (Int(maximum(entry)) + Δ + 1):(Int(minimum(newexit)) - 1)
            length(strsites) == nstr || invariant_violation(
                "expansion of a channel: the gap has $(length(strsites)) site(s), expected $nstr"
            )
            _emit_stretched!(d, entry .+ Δ, newexit, ops, bonds, ne, entrybond, strsites, strpairs, w)
            nstr += P
        end
    end
    return d
end

# One stretched translate: a composite string distributes over its letters at every gap site.
function _emit_stretched!(
        d::Dictionary{Support{I}, ComplexF64}, entry, exit_, ops, bonds, ne, entrybond,
        strsites, strpairs, w
    ) where {I}
    nstr = length(strsites)
    active = filter(p -> !ispassthrough(p[1]), strpairs)
    if isempty(active) || nstr == 0
        # all gap sites idle: one term, with the string's scalar part once per site
        scalar = sum(c for (letter, c) in strpairs if ispassthrough(letter); init = zero(ComplexF64))
        (nstr == 0 || !iszero(scalar)) || return d
        wtot = nstr == 0 ? w : w * scalar^nstr
        newsites = Int[entry..., exit_...]
        newops = Letter{I}[ops...]
        newbonds = eltype(bonds)[bonds...]
        _insert_stretched!(d, newsites, newops, newbonds, wtot)
        return d
    end
    for combo in Iterators.product(ntuple(_ -> strpairs, nstr)...)
        wtot = w
        strs = Tuple{Int, Letter{I}}[]
        for (s, (letter, c)) in zip(strsites, combo)
            wtot *= c
            ispassthrough(letter) || push!(strs, (s, letter))
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
