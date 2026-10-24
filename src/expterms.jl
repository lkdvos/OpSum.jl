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
        d::Dictionary{Term{I}, ComplexF64}, k::ExpKey{I}, coeff, P::Int, N::Int
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
        d::Dictionary{Term{I}, ComplexF64}, entry, exit_, ops, bonds, ne, entrybond,
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
        d::Dictionary{Term{I}, ComplexF64}, newsites, newops, newbonds, w
    ) where {I}
    # multiplicity-free fusion: every vertex label is 1
    keys_ = Factor{I}[Factor{I}(o, b) for (o, b) in zip(newops, newbonds)]
    # `Term`'s `==`/`hash` ignore the coefficient, so `setwith!` sums coincident terms
    setwith!(+, d, Term{I}(newsites, keys_, one(ComplexF64)), ComplexF64(w))
    return d
end

# Every translate of every channel inside `1:N` as an ordinary bag; exponential in the gap (oracle).
function expand_channels(es::ExpSum{I}, P::Int, N::Int) where {I}
    d = Dictionary{Term{I}, ComplexF64}()
    for (k, v) in pairs(es.channels)
        _expand_channel!(d, k, v, P, N)
    end
    filter!(!iszero, d)
    return Terms{I}([Term{I}(t.sites, t.factors, v) for (t, v) in pairs(d)])
end

# --- lowering to a weighted automaton -------------------------------------------------------------
#
# A channel becomes an automaton whose states are its suffix classes (a right vertex is a
# `(channel, state)` pair): E_b entry block partly placed; W_δ mandatory wait before the first legal
# exit; D_δ the cyclic part (δ = distance to the next exit); X_b exit block partly placed. Names are
# interned so that name equality ⟺ class equality: `0` is the exhausted class, one id per loop
# descriptor, and acyclic chains cons `(key, successor name)`.

# `minremain`: fewest further sites to complete (the window boundary prunes on it); `cyclic`: in the
# loop core; `trans`: `(key at the next site, target state, weight)`, target `0` = exhausted.
struct ChannelState{I <: Sector}
    name::Int
    bond::I
    minremain::Int
    cyclic::Bool
    trans::Vector{Tuple{Factor{I}, Int, ComplexF64}}
end

# A lowered `ExpKey`; `coeff` rides on the entry edge so channels differing only there share a class.
struct ExpChannel{I <: Sector}
    phase::Int
    period::Int
    entrykey::Factor{I}
    coeff::ComplexF64
    span::Int
    start::Int
    states::Vector{ChannelState{I}}
end

# Largest site of `1:N` at which the channel may still enter (0 if none).
function lastentry(c::ExpChannel, N::Int)
    hi = N - c.span
    hi < 1 && return 0
    s = hi - mod(hi - c.phase, c.period)
    return s < 1 ? 0 : s
end

isentry(c::ExpChannel, s::Int, N::Int) = mod(s - c.phase, c.period) == 0 && s + c.span <= N

# shared across all channels of a model so that classes which must merge are equal
mutable struct _ChannelNames{I <: Sector}
    cons::Dictionary{Tuple{Factor{I}, Int}, Int}
    loop::Dictionary{Tuple{ComplexF64, Vector{Tuple{Factor{I}, ComplexF64}}, Factor{I}, Int, Int, Int}, Int}
    n::Int
end
_ChannelNames{I}() where {I} = _ChannelNames{I}(Dictionary(), Dictionary(), 0)

function _intern!(d, names::_ChannelNames, k)
    id = get(d, k, 0)
    iszero(id) || return id
    names.n += 1
    insert!(d, k, names.n)
    return names.n
end
_cons!(names::_ChannelNames{I}, key::Factor{I}, succ::Int) where {I} =
    _intern!(names.cons, names, (key, succ))
_loopname!(names::_ChannelNames{I}, λ, strs, exitkey, exitname, P, δ) where {I} =
    _intern!(names.loop, names, (λ, strs, exitkey, exitname, P, δ))

# Content-derived order, so interned ids (hence bond order) do not depend on the order channels were written.
function _channelorder(k::ExpKey)
    return (
        Int.(k.term.sites),
        Int(k.exitsite),
        [(key.letter.c, key.letter.n) for key in k.term.factors],
        [key.bond for key in k.term.factors],
        (real(k.decay), imag(k.decay)),
        [(l === nothing, l === nothing ? 0 : l.n, real(c), imag(c)) for (l, c) in k.string],
    )
end

function _lower_channels(es::ExpSum{I}, P::Int) where {I}
    P >= 1 || throw(ArgumentError("channel period must be positive, got $P"))
    names = _ChannelNames{I}()
    entries = sort!(collect(pairs(es.channels)); by = kv -> _channelorder(first(kv)))
    return ExpChannel{I}[_lower_channel(k, ComplexF64(v), P, names) for (k, v) in entries]
end

# Reverse acyclic chain over `range` (descending bonds); at `b+1 == top` it steps to `boundary_target`.
function _build_acyclic_chain!(
        states::Vector{ChannelState{I}}, names::_ChannelNames{I}, bondat, stepkey, slast::Int,
        range, top::Int, boundary_target::Int
    ) where {I}
    idx = Dictionary{Int, Int}()
    for b in range
        bond = bondat(b)
        nkey = stepkey(b + 1, bond)
        target = b + 1 == top ? boundary_target : idx[b + 1]
        name = _cons!(names, nkey, iszero(target) ? 0 : states[target].name)
        push!(
            states, ChannelState{I}(
                name, bond, slast - b, false,
                Tuple{Factor{I}, Int, ComplexF64}[(nkey, target, one(ComplexF64))]
            )
        )
        insert!(idx, b, length(states))
    end
    return idx
end

function _lower_channel(
        k::ExpKey{I}, coeff::ComplexF64, P::Int, names::_ChannelNames{I}
    ) where {I}
    sites = k.term.sites
    keys_ = k.term.factors
    K = length(sites)
    ne = count(<(k.exitsite), sites)
    slast = sites[K]
    sexit = sites[ne + 1]
    δ0 = sexit - sites[ne]
    nloop = max(P, δ0)
    cmid = keys_[ne].bond               # running charge along the string

    states = ChannelState{I}[]
    # the key applied at site `nb`, and the running charge just left of it
    function stepkey(nb::Int, bond::I)
        j = findfirst(==(nb), sites)
        return j === nothing ? Factor{I}(passthrough(I), bond) : keys_[j]
    end
    bondat(b::Int) = keys_[count(<=(b), sites)].bond

    # exit block: acyclic chain down to the exhausted class
    xidx = _build_acyclic_chain!(states, names, bondat, stepkey, slast, (slast - 1):-1:sexit, slast, 0)

    # the loop: transitions are filled in a second pass (they are cyclic)
    exittarget = ne + 1 == K ? 0 : xidx[sexit]
    exitname = iszero(exittarget) ? 0 : states[exittarget].name
    exitkey = keys_[ne + 1]
    # λ rides as an edge weight so the entry carries a letter (a letter-less `SiteOperator` is dropped
    # at tensor assembly); the `cyclic` loop states are forced into the cover, else λ accumulates.
    strtrans = Tuple{Factor{I}, ComplexF64}[
        (Factor{I}(l === nothing ? passthrough(I) : l, cmid), k.decay * c) for (l, c) in k.string
    ]
    loopidx = zeros(Int, nloop)
    for δ in 1:nloop
        name = _loopname!(names, k.decay, strtrans, exitkey, exitname, P, δ)
        push!(
            states, ChannelState{I}(
                name, cmid, δ + (slast - sexit), δ <= P, Tuple{Factor{I}, Int, ComplexF64}[]
            )
        )
        loopidx[δ] = length(states)
    end
    for δ in 1:nloop
        trans = states[loopidx[δ]].trans
        target = δ > 1 ? loopidx[δ - 1] : loopidx[P]
        for (key, w) in strtrans
            push!(trans, (key, target, w))
        end
        δ == 1 && push!(trans, (exitkey, exittarget, one(ComplexF64)))
    end

    # entry block: acyclic chain into the loop
    eidx = _build_acyclic_chain!(
        states, names, bondat, stepkey, slast, (sites[ne] - 1):-1:sites[1], sites[ne], loopidx[δ0]
    )

    start = ne == 1 ? loopidx[δ0] : eidx[sites[1]]
    return ExpChannel{I}(
        mod1(sites[1], P), P, keys_[1], coeff, slast - sites[1], start, states
    )
end
