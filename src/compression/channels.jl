# Lowering of decaying channels (`ExpSum`) to a weighted automaton the sweep consumes.

using TensorKit: Sector

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
