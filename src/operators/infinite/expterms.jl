# Exponentially decaying interactions `Σ_{i<j} λ^{j-i-1} A_i S_{i+1}…S_{j-1} B_j`, `|λ| < 1`, as
# `channels` the sweep lowers to a single bond index carrying `λ` on its diagonal (design:
# `research/infinite-mpo.md` §7). A channel is one fully specified representative term plus the site
# where its exit block starts:
#
#     expterm(SS[1, 2]; decay = 0.5)
#     expterm(couple(couple(A[1], A′[2]; to = c), B[4]); decay = 0.5, exitsite = 4)
#
# With lattice period `P` (`L` for an `InfiniteChain`, `1` for a finite chain) it represents every
# translate by multiples of `P` that fits, exit block pushed right; a translate weighs
# `coeff · λ^{string sites}`. The string, if given, sits only on the stretched gap.

using TensorKit: Sector, unit
using .IrrepTensorOperators: IrrepOperator

export expterm

# --- the descriptor -------------------------------------------------------------------------------

"""
    ExpKey{I<:Sector}

One decaying interaction as a hashable descriptor: the representative `term` (a fully coupled
[`Term`](@ref)), the `exitsite` where its exit block starts (factors at sites `< exitsite` are the
entry block, the rest the exit block; both non-empty), the `decay` `λ`, and the on-site `string`
carried by the stretched gap, as a canonical `(letter, coefficient)` list (`nothing` = identity).
"""
struct ExpKey{I <: Sector}
    term::Term{I}
    exitsite::Int
    decay::ComplexF64
    string::Vector{Tuple{Union{Nothing, IrrepOperator{I}}, ComplexF64}}
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

"""
    channelspan(k::ExpKey) -> Int

Span of the channel's *shortest* translate (its range is otherwise unbounded).
"""
channelspan(k::ExpKey) = Int(maximum(k.term.sites) - minimum(k.term.sites))

# Shift a channel by `Δ` sites: the representative and the exit site move together.
function translate(k::ExpKey{I}, Δ) where {I}
    iszero(Δ) && return k
    term = Term{I}(k.term.sites .+ Int(Δ), k.term.keys, k.term.coeff)
    return ExpKey{I}(term, k.exitsite + Int(Δ), k.decay, k.string)
end

# --- the containers -------------------------------------------------------------------------------

"""
    ExpSum{I<:Sector}

A sum of exponentially decaying interactions ([`ExpKey`](@ref) descriptors with coefficients), the
geometric counterpart of [`Terms`](@ref). Built by [`expterm`](@ref) and added to an
[`OperatorSum`](@ref): `opsum(lat, h, expterm(h; decay = λ))`. A sum of exponentials is a sum of
channels; each distinct `λ` costs its own bond index.
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

# Arithmetic among channels. Meeting a `Terms` bag happens in an `OperatorSum`.
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

# Flatten the string operator into a canonical `(letter, coefficient)` list. Letters must be neutral:
# a charged one would make the running bond charge drift, so the loop would not be a *self*-loop.
function _string_terms(::Type{I}, op) where {I <: Sector}
    A = IrrepOperator{I}
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

An exponentially decaying interaction generated from the representative term(s) `t` by stretching the
gap just before `exitsite` geometrically, with ratio `decay` per site:

```julia
SS = spin_ops(SU2Irrep).S_exchange
expterm(SS[1, 2]; decay = 0.5)                 # Σ_{i<j} 0.5^{j-i-1} S⃗_i·S⃗_j
expterm(SS[1, 3]; decay = 0.5)                 # the same, but never closer than next-nearest neighbours
```

`t` must be fully coupled with at least two active sites (it carries the fusion tree); a composite
operand yields one channel per term. `exitsite` must be an active site of the representative and not
its first. `string` is a charge-neutral `SiteOperator` carried by every site of the stretched gap
(default: the identity). `decay` must satisfy `0 < |λ| < 1`. The lattice supplies the translation
period: `L` on an [`InfiniteChain`](@ref), one site on a finite chain.
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
        k = Term{I}(term.sites, term.keys, one(ComplexF64))
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

# Write every translate of channel `k` that fits inside `1:N` (period `P`) into `d` as ordinary terms:
# the oracle for the sweep, which never sees it. Entry anchors run over the sites of `1:N` congruent to
# the representative's, and for each the exit block is pushed right in steps of `P` until it leaves.
function _expand_channel!(
        d::Dictionary{Term{I}, ComplexF64}, k::ExpKey{I}, coeff, P::Int, N::Int
    ) where {I}
    sites = k.term.sites
    ops = [key.op for key in k.term.keys]
    bonds = [key.bond for key in k.term.keys]
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

# One stretched translate. A composite string distributes over its letters at *every* gap site, so
# the general case is a product over the gap (`nothing` letters stay idle).
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
        newops = IrrepOperator{I}[ops...]
        newbonds = eltype(bonds)[bonds...]
        _insert_stretched!(d, newsites, newops, newbonds, wtot)
        return d
    end
    for combo in Iterators.product(ntuple(_ -> string, nstr)...)
        wtot = w
        strs = Tuple{Int, IrrepOperator{I}}[]
        for (s, (letter, c)) in zip(strsites, combo)
            wtot *= c
            letter === nothing || push!(strs, (s, letter))
        end
        iszero(wtot) && continue
        newsites = Int[entry..., first.(strs)..., exit_...]
        newops = IrrepOperator{I}[ops[1:ne]..., last.(strs)..., ops[(ne + 1):end]...]
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
    keys_ = ITOKey{I}[ITOKey{I}(o, b, 1) for (o, b) in zip(newops, newbonds)]
    # `Term`'s `==`/`hash` ignore the coefficient, so `setwith!` sums coincident terms
    setwith!(+, d, Term{I}(newsites, keys_, one(ComplexF64)), ComplexF64(w))
    return d
end

"""
    expand_channels(es::ExpSum, P::Int, N::Int) -> Terms

Every translate of every channel that fits inside `1:N`, as an ordinary bag. Exponential in the
string's letters times the gap: a test/oracle tool.
"""
function expand_channels(es::ExpSum{I}, P::Int, N::Int) where {I}
    d = Dictionary{Term{I}, ComplexF64}()
    for (k, v) in pairs(es.channels)
        _expand_channel!(d, k, v, P, N)
    end
    filter!(!iszero, d)
    return Terms{I}([Term{I}(t.sites, t.keys, v) for (t, v) in pairs(d)])
end

# --- lowering to a weighted automaton -------------------------------------------------------------
#
# What the sweep consumes. A channel becomes a small automaton whose states are exactly the suffix
# classes it can occupy, so a right vertex of the graph is a `(channel, state)` pair:
#
#   E_b   entry block partly placed (one state per bond of the entry block)   → next factor / idle
#   W_δ   mandatory wait before the first legal exit (δ > period)             → string, weight λ·c
#   D_δ   the CYCLIC part, δ = period … 1 (δ = distance to the next exit)     → string, and at δ = 1
#                                                                              also the exit factor
#   X_b   exit block partly placed                                           → next factor / idle
#
# `δ` is the distance to the next legal exit, so the phase alignment a period `> 1` needs lives inside
# the state. Every state carries an interned name with `name equality ⟺ class equality`, which the
# per-bond suffix merge and the canonical order are keyed on. Hash-consing does it once the cycle is
# cut: `0` names the exhausted class, one id names each *loop descriptor* `(λ, string transitions, exit
# key, exit-class name, period, δ)` (which fixes the whole cyclic future), and the acyclic `E`/`X`
# chains cons `(key at the next site, successor name)` on top, as `_rel_suffix_ids` does for shapes.

# One state of a lowered channel's automaton: interned class `name`, running bond charge `bond`,
# `minremain` (fewest further sites needed to complete, what the window boundary prunes on), whether
# it is in the `cyclic` core (forced into the bond basis, see `_vc_component`), and its transitions
# `(key applied at the next site, target state, weight)`; target `0` is the exhausted class.
struct ChannelState{I <: Sector}
    name::Int
    bond::I
    minremain::Int
    cyclic::Bool
    trans::Vector{Tuple{ITOKey{I}, Int, ComplexF64}}
end

# A lowered [`ExpKey`](@ref): the automaton plus what the sweep needs to enter it — the anchor `phase`
# (mod `period`), the `entrykey` placed there, the `coeff` (carried by the entry edge, so channels
# differing only in their entry are the same class from then on) and the representative's `span`.
struct ExpChannel{I <: Sector}
    phase::Int
    period::Int
    entrykey::ITOKey{I}
    coeff::ComplexF64
    span::Int
    start::Int
    states::Vector{ChannelState{I}}
end

# Largest site of `1:N` at which this channel may still enter (0 if none): the anchor must match the
# phase and the shortest translate must still fit.
function lastentry(c::ExpChannel, N::Int)
    hi = N - c.span
    hi < 1 && return 0
    s = hi - mod(hi - c.phase, c.period)
    return s < 1 ? 0 : s
end

isentry(c::ExpChannel, s::Int, N::Int) = mod(s - c.phase, c.period) == 0 && s + c.span <= N

# name interning, shared across all channels of one model so that classes which must merge are equal
mutable struct _ChannelNames{I <: Sector}
    cons::Dictionary{Tuple{ITOKey{I}, Int}, Int}
    loop::Dictionary{Tuple{ComplexF64, Vector{Tuple{ITOKey{I}, ComplexF64}}, ITOKey{I}, Int, Int, Int}, Int}
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
_cons!(names::_ChannelNames{I}, key::ITOKey{I}, succ::Int) where {I} =
    _intern!(names.cons, names, (key, succ))
_loopname!(names::_ChannelNames{I}, λ, strs, exitkey, exitname, P, δ) where {I} =
    _intern!(names.loop, names, (λ, strs, exitkey, exitname, P, δ))

# Content-derived order, so interned ids (hence bond order) do not depend on the order channels were written.
function _channelorder(k::ExpKey)
    return (
        Int.(k.term.sites),
        Int(k.exitsite),
        [(key.op.c, key.op.n) for key in k.term.keys],
        [key.bond for key in k.term.keys],
        (real(k.decay), imag(k.decay)),
        [(l === nothing, l === nothing ? 0 : l.n, real(c), imag(c)) for (l, c) in k.string],
    )
end

# Lower every channel of `es` to its automaton, with names interned across the list. `P` is the
# lattice period (the cell length, or 1 on a finite chain).
function _lower_channels(es::ExpSum{I}, P::Int) where {I}
    P >= 1 || throw(ArgumentError("channel period must be positive, got $P"))
    names = _ChannelNames{I}()
    entries = sort!(collect(pairs(es.channels)); by = kv -> _channelorder(first(kv)))
    return ExpChannel{I}[_lower_channel(k, ComplexF64(v), P, names) for (k, v) in entries]
end

# Build a reverse acyclic chain of states over `range` (descending bond positions), each stepping to
# the previously built one; at the top (`b+1 == top`) it steps to `boundary_target` instead.
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
                Tuple{ITOKey{I}, Int, ComplexF64}[(nkey, target, one(ComplexF64))]
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
    keys_ = k.term.keys
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
        return j === nothing ? ITOKey{I}(passthrough(I), bond, 1) : keys_[j]
    end
    bondat(b::Int) = keys_[count(<=(b), sites)].bond

    # exit block: acyclic chain down to the exhausted class
    xidx = _build_acyclic_chain!(states, names, bondat, stepkey, slast, (slast - 1):-1:sexit, slast, 0)

    # the loop: transitions are filled in a second pass (they are cyclic; the names are not)
    exittarget = ne + 1 == K ? 0 : xidx[sexit]
    exitname = iszero(exittarget) ? 0 : states[exittarget].name
    exitkey = keys_[ne + 1]
    # Load-bearing: λ rides as an edge *weight* on a letter-carrying key, because tensor assembly
    # iterates `pairs(op)` and a letter-less `SiteOperator` would be silently dropped. The loop states
    # are `cyclic` and so are forced into the vertex cover, else λ would accumulate instead of landing
    # on the diagonal.
    strtrans = Tuple{ITOKey{I}, ComplexF64}[
        (ITOKey{I}(l === nothing ? passthrough(I) : l, cmid, 1), k.decay * c) for (l, c) in k.string
    ]
    loopidx = zeros(Int, nloop)
    for δ in 1:nloop
        name = _loopname!(names, k.decay, strtrans, exitkey, exitname, P, δ)
        push!(
            states, ChannelState{I}(
                name, cmid, δ + (slast - sexit), δ <= P, Tuple{ITOKey{I}, Int, ComplexF64}[]
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
