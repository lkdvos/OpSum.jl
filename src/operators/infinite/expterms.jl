# Exponentially decaying interactions: `Σ_{i<j} λ^{j-i-1} A_i S_{i+1}…S_{j-1} B_j`, `|λ| < 1` — an
# infinite family of terms, represented as `channels` that the sweep lowers to a single bond index
# carrying `λ` on its diagonal (see `research/infinite-mpo.md` §7 and `_lower_channels` below).
#
# A channel is one fully specified representative term plus the site where its exit block starts:
#
#     expterm(dot(S[1], S[2]); decay = 0.5)
#     expterm(couple(couple(A[1], A′[2]; to = c), B[4]); decay = 0.5, exitsite = 4)
#
# Given lattice period `P` (`L` for an `InfiniteChain`, `1` for a finite chain), a channel represents
# every translate by multiples of `P` whose support fits the lattice, exit block pushed right along
# with it; a translate's weight is `coeff · λ^{string sites}`. So `λ` counts per site: the shortest
# translate carries `λ^g` (`g` = the representative's own gap) and every extra period costs `λ^P`. The
# string operator, if given, sits only on the stretched gap.

using TensorKit: Sector, unit
using .IrrepTensorOperators: IrrepOperator

export expterm

# --- the descriptor -------------------------------------------------------------------------------

"""
    ExpKey{I<:Sector}

One exponentially decaying interaction, as a hashable descriptor: the representative `term` (a fully
coupled [`Term`](@ref)), the site `exitsite` at which its exit block starts, the `decay` `λ`, and
the on-site `string` operator carried by the stretched gap, flattened to a canonical
`(letter, coefficient)` list (a `nothing` letter is the pass-through identity).

The entry block is the representative's factors at sites `< exitsite`, the exit block those at sites
`>= exitsite`; both are non-empty.
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
    _ispassthroughstring(k) || print(io, ", string=", k.string)
    return print(io, ")")
end

# entry block = factors strictly left of the exit site; exit block = the rest. Both non-empty by
# construction (`expterm` validates it).
entrysites(k::ExpKey) = filter(<(k.exitsite), k.term.sites)
exitsites(k::ExpKey) = filter(>=(k.exitsite), k.term.sites)

"""
    channelspan(k::ExpKey) -> Int

Span of the channel's *shortest* translate: the number of bonds between the first and last factor of
the representative. The interaction range is unbounded; this is the range the sweep has to reach its
periodic fixed point over.
"""
channelspan(k::ExpKey) = Int(maximum(k.term.sites) - minimum(k.term.sites))

# Number of string sites in the shortest translate (`≥ 0`).
stringgap(k::ExpKey) = Int(k.exitsite - maximum(entrysites(k))) - 1

_ispassthroughstring(k::ExpKey) = length(k.string) == 1 && only(k.string)[1] === nothing &&
    isone(only(k.string)[2])

"""
    translate(k::ExpKey, Δ) -> ExpKey

Shift a channel by `Δ` sites: the representative and the exit site move together, everything else is
carried over untouched.
"""
function translate(k::ExpKey{I}, Δ) where {I}
    iszero(Δ) && return k
    term = Term{I}(k.term.sites .+ Int(Δ), k.term.keys, k.term.coeff)
    return ExpKey{I}(term, k.exitsite + Int(Δ), k.decay, k.string)
end

# --- the containers -------------------------------------------------------------------------------

"""
    ExpSum{I<:Sector}

A sum of exponentially decaying interactions: [`ExpKey`](@ref) descriptors with coefficients, the
geometric counterpart of [`Terms`](@ref). Built by [`expterm`](@ref); `+` with a `Terms` bag gives a
[`MixedSum`](@ref), which is what `irrep_mpo` consumes.

Summing several channels is how a *sum of exponentials* is written (each keeps its own `λ`, and
channels with different `λ` stay linearly independent, so each costs its own bond index).
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

"""
    MixedSum{I<:Sector}

A Hamiltonian with both finite-range terms and exponentially decaying ones: a [`Terms`](@ref) bag plus
an [`ExpSum`](@ref). Produced by adding the two, and accepted by `irrep_mpo` on an
[`InfiniteChain`](@ref) or with an explicit `sites` vector.

Latticeless, like the `Terms` it is built from: on an infinite chain the lattice comes from the chain,
and the finite entry point takes its `sites` argument.

```julia
H = dot(S[1], S[2]) + expterm(dot(S[1], S[2]); decay = 0.4)
```
"""
struct MixedSum{I <: Sector}
    terms::Terms{I}
    channels::ExpSum{I}
end

function Base.show(io::IO, H::MixedSum)
    return print(io, "MixedSum(", H.terms, ", ", H.channels, ")")
end

# a model may be nothing but channels, or nothing but terms
MixedSum(t::Terms{I}) where {I} = MixedSum{I}(t, ExpSum{I}())
MixedSum(e::ExpSum{I}) where {I} = MixedSum{I}(Terms{I}(), e)
MixedSum(H::MixedSum) = H

# Arithmetic: `+` closes over Terms / ExpSum / MixedSum, so a model can be written as one sum.
function Base.:+(a::ExpSum{I}, b::ExpSum{I}) where {I}
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for (k, v) in pairs(a.channels)
        setwith!(+, d, k, ComplexF64(v))
    end
    for (k, v) in pairs(b.channels)
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

# Coefficients are `ComplexF64` throughout now, so there is no promotion left to do.
_mixed(t::Terms{I}, e::ExpSum{I}) where {I} = MixedSum{I}(t, e)

Base.:+(a::Terms{I}, b::ExpSum{I}) where {I} = _mixed(a, b)
Base.:+(a::ExpSum{I}, b::Terms{I}) where {I} = _mixed(b, a)
Base.:+(a::MixedSum{I}, b::Terms{I}) where {I} = _mixed(a.terms + b, a.channels)
Base.:+(a::Terms{I}, b::MixedSum{I}) where {I} = _mixed(a + b.terms, b.channels)
Base.:+(a::MixedSum{I}, b::ExpSum{I}) where {I} = _mixed(a.terms, a.channels + b)
Base.:+(a::ExpSum{I}, b::MixedSum{I}) where {I} = _mixed(b.terms, a + b.channels)
Base.:+(a::MixedSum{I}, b::MixedSum{I}) where {I} =
    _mixed(a.terms + b.terms, a.channels + b.channels)

VectorInterface.scale(a::MixedSum, α::Number) = _mixed(scale(a.terms, α), scale(a.channels, α))
Base.:*(α::Number, a::MixedSum) = scale(a, α)
Base.:*(a::MixedSum, α::Number) = scale(a, α)
Base.:/(a::MixedSum, α::Number) = scale(a, inv(α))
Base.:-(a::MixedSum) = scale(a, -1)
Base.:-(a::MixedSum, b::Union{Terms, ExpSum, MixedSum}) = a + (-b)
Base.:-(a::Union{Terms, ExpSum}, b::MixedSum) = a + (-b)
Base.:-(a::Terms{I}, b::ExpSum{I}) where {I} = a + (-b)
Base.:-(a::ExpSum{I}, b::Terms{I}) where {I} = a + (-b)

# --- the constructor ------------------------------------------------------------------------------

# Flatten the string operator into a canonical `(letter, coefficient)` list. Every letter must carry
# the trivial charge: the running bond charge of a channel is fixed along its string (that is what
# makes the loop a *self*-loop), and a charged string letter would make it drift.
function _string_terms(::Type{I}, op) where {I <: Sector}
    A = IrrepOperator{I}
    terms = if op === nothing
        Tuple{Union{Nothing, A}, ComplexF64}[(nothing, one(ComplexF64))]
    else
        Tuple{Union{Nothing, A}, ComplexF64}[
            (letter, ComplexF64(c)) for (letter, c) in pairs(op) if !iszero(c)
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

An exponentially decaying interaction, generated from the representative term(s) `t` by stretching
the gap just before `exitsite` geometrically with ratio `decay` per site:

```julia
expterm(dot(S[1], S[2]); decay = 0.5)          # Σ_{i<j} 0.5^{j-i-1} S_i·S_j
expterm(couple(Sp[1], Sm[3]); decay = 0.5)     # the same, but never closer than third neighbours
```

`t` must be a fully coupled term sum whose terms have at least two active sites — it carries the
caterpillar fusion tree, so all fusion channels are already named. A composite operand (e.g. `Sᶻ`, or
an `xxz` bond written as three terms) yields one channel per term, exactly as `couple` distributes.

`exitsite` must be one of the representative's active sites and not its first: the factors left of it
are the *entry block*, the rest the *exit block*, and it is the gap between the two that is stretched.
`string` is an on-site operator (a `SiteOperator`) carried by every site of that gap; it must be
charge-neutral. The default is the bare pass-through identity.

The lattice supplies the translation period: on an [`InfiniteChain`](@ref) of `L` sites the entry and
exit both step by `L`, on a finite chain by one site. `decay` counts per *site* of the string, and
must satisfy `0 < |λ| < 1` — `λ = 1` is not summable.

Add the result to a `Terms` bag to get a [`MixedSum`](@ref), which is what `irrep_mpo` consumes.
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
    isone(λ) && throw(
        ArgumentError("expterm: decay = 1 is not summable (Σ_r λ^r diverges); need |λ| < 1")
    )
    abs(λ) < 1 || throw(
        ArgumentError("expterm: need |decay| < 1 for a summable interaction, got |λ| = $(abs(λ))")
    )
    isempty(t.terms) && throw(ArgumentError("expterm: the representative has no terms"))
    str = _string_terms(I, string)

    d = Dictionary{ExpKey{I}, ComplexF64}()
    for term in t.terms
        # the coefficient rides in the `ExpSum` dictionary, so the stored representative is
        # normalised to 1 — otherwise it would be counted twice on expansion
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

expterm(t::MixedSum; kwargs...) = throw(
    ArgumentError("expterm: the representative must be an ordinary (finite-range) Terms bag")
)

# --- explicit expansion (the oracle) --------------------------------------------------------------

"""
    _expand_channel!(d, k::ExpKey, coeff, P::Int, N::Int)

Write every translate of channel `k` whose support fits inside `1:N` into the term dictionary `d`,
with period `P`. This is the channel's meaning spelled out as ordinary terms, and it is what both the
faithfulness oracle and [`window_terms`](@ref) use — the sweep never sees it.

The number of translates is finite because the window is: entry anchors run over the sites of `1:N`
congruent to the representative's, and for each the exit block is pushed right in steps of `P` until
it leaves the window.
"""
function _expand_channel!(
        d::Dictionary{Term{I}, ComplexF64}, k::ExpKey{I}, coeff, P::Int, N::Int
    ) where {I}
    sites = k.term.sites
    # a `Term` stores one `ITOKey` per factor; its `bond` is that position's running caterpillar charge
    ops = [key.op for key in k.term.keys]
    bonds = [key.bond for key in k.term.keys]
    ne = count(<(k.exitsite), sites)                     # size of the entry block
    entry, exit_ = sites[1:ne], sites[(ne + 1):end]
    entrybond = bonds[ne]                                # running charge along the string
    g0 = stringgap(k)
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

# One stretched translate. With a bare pass-through string this is a single term; a composite string
# distributes over its letters at *every* string site, so the general case is a product over the gap
# (`nothing` letters stay idle and contribute no active site).
function _emit_stretched!(
        d::Dictionary{Term{I}, ComplexF64}, entry, exit_, ops, bonds, ne, entrybond,
        strsites, string, w
    ) where {I}
    nstr = length(strsites)
    active = filter(p -> p[1] !== nothing, string)
    if isempty(active) || nstr == 0
        # every string site is a bare pass-through: one term, and the coefficient picks up the
        # (letter-less) scalar part of the string once per site
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
    # a `Term` stores one `ITOKey` per factor — letter, running caterpillar bond charge, vertex label
    # — rather than a fusion tree; multiplicity-free fusion means every vertex label is 1
    keys_ = ITOKey{I}[ITOKey{I}(o, b, 1) for (o, b) in zip(newops, newbonds)]
    # `Term`'s `==`/`hash` ignore the coefficient, so `setwith!` accumulates coincident terms
    setwith!(+, d, Term{I}(newsites, keys_, one(ComplexF64)), ComplexF64(w))
    return d
end

"""
    expand_channels(es::ExpSum, P::Int, N::Int) -> Terms

Every translate of every channel of `es` that fits inside `1:N`, as an ordinary term bag. Exponential
in the string's number of letters times the gap, so this is a test/oracle tool, not a build step.
"""
function expand_channels(es::ExpSum{I}, P::Int, N::Int) where {I}
    d = Dictionary{Term{I}, ComplexF64}()
    for (k, v) in pairs(es.channels)
        _expand_channel!(d, k, v, P, N)
    end
    filter!(!iszero, d)
    return Terms{I}([Term{I}(t.sites, t.keys, v) for (t, v) in pairs(d)])
end

"""
    chain_terms(H::MixedSum, N::Int) -> Terms

The literal term sum a [`MixedSum`](@ref) stands for on a finite chain of `N` sites: the finite-range
terms as written, plus every translate (period 1) of every channel that fits. This is what the finite
`irrep_mpo(H::MixedSum, sites)` represents.
"""
function chain_terms(H::MixedSum{I}, N::Int) where {I}
    return H.terms + expand_channels(H.channels, 1, N)
end

# --- lowering to a weighted automaton -------------------------------------------------------------
#
# What the sweep consumes. A channel becomes a small automaton whose states are exactly the suffix
# classes it can occupy, so a right vertex of the graph is a `(channel, state)` pair and everything the
# sweep needs — the on-site key, the successor class, the weight — is a table lookup:
#
#   E_b   entry block partly placed (one state per bond of the entry block)   → next factor / idle
#   W_δ   mandatory wait before the first legal exit (δ > period)             → string, weight λ·c
#   D_δ   the CYCLIC part, δ = period … 1 (δ = distance to the next exit)     → string, and at δ = 1
#                                                                              also the exit factor
#   X_b   exit block partly placed                                           → next factor / idle
#
# `δ` is the distance to the next legal exit, so the phase alignment a period `> 1` needs lives
# *inside the state* and the sweep does no phase arithmetic; a representative whose gap exceeds the
# period simply gets a `W` chain in front of the cycle.
#
# **Names.** Class identity is what the per-bond suffix-merge and the canonical order are keyed on, so
# every state carries an interned name with `name equality ⟺ class equality`. Bottom-up hash-consing
# does it, once the cycle is cut: `0` names the exhausted class, one reserved id names each *loop
# descriptor* `(λ, string transitions, exit key, exit-class name, period, δ)` — which determines the
# whole cyclic future in closed form — and the acyclic `E`/`X` chains cons `(key at the next site,
# successor name)` on top, exactly as `_rel_suffix_ids` conses shapes. Names are relative to the bond,
# hence position-independent, hence usable for the merge *and* for the translation-invariant order.
# This is what replaces the partition refinement `research/infinite-mpo.md` §7 expected to need.

"""
    ChannelState{I}

One state of a lowered channel's automaton: its interned class `name`, the running bond `bond` charge
while in it, `minremain` (the fewest further sites the class needs to complete — what the finite/window
boundary prunes on), whether it is part of the `cyclic` core (those must be forced into the bond basis,
see `_vc_component`), and its transitions `(key applied at the next site, target state, weight)`. A
target of `0` is the exhausted class.
"""
struct ChannelState{I <: Sector}
    name::Int
    bond::I
    minremain::Int
    cyclic::Bool
    trans::Vector{Tuple{ITOKey{I}, Int, ComplexF64}}
end

"""
    ExpChannel{I}

A lowered [`ExpKey`](@ref): the automaton above plus what the sweep needs to *enter* it — the entry
`phase` (the anchor site mod `period`), the `entrykey` placed there, the channel `coeff` (the whole
coefficient rides on the entry edge, so channels differing only in their entry are literally the same
class from then on) and the representative's `span`.
"""
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

# Content-derived order, so the interned ids — and hence the bond index order downstream — do not
# depend on the order the channels were written in (the same property `test_infinite_graph.jl` pins
# for terms).
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

"""
    _lower_channels(es::ExpSum, P::Int) -> Vector{ExpChannel}

Lower every channel of `es` to its automaton, with names interned across the whole list. `P` is the
lattice period (the unit-cell length, or 1 on a finite chain).
"""
function _lower_channels(es::ExpSum{I}, P::Int) where {I}
    P >= 1 || throw(ArgumentError("channel period must be positive, got $P"))
    names = _ChannelNames{I}()
    entries = sort!(collect(pairs(es.channels)); by = kv -> _channelorder(first(kv)))
    return ExpChannel{I}[_lower_channel(k, ComplexF64(v), P, names) for (k, v) in entries]
end

# Build a reverse acyclic chain of `ChannelState`s over `range` (a descending `hi:-1:lo` run of bond
# positions), each state stepping to the previously built one; at the top of the range (`b+1 == top`)
# it steps instead to `boundary_target`. Shared by `_lower_channel`'s entry- and exit-block loops,
# which differ only in the range walked and in what lies past its top.
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
    # a `Term` stores the active `(site, ITOKey)` factors directly, in site order
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

    # the loop: transitions are filled in a second pass because they are cyclic; the *names* are not,
    # which is the whole point of naming a loop by its descriptor
    exittarget = ne + 1 == K ? 0 : xidx[sexit]
    exitname = iszero(exittarget) ? 0 : states[exittarget].name
    exitkey = keys_[ne + 1]
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
