# Reduced-MPO sweeps for the term automaton. `VertexCover`/`SequentialSVD` run the persistent
# bipartite-graph sweep here (cost `Θ(M·K) + Θ(Σ_terms span)`); `IndependentSVD` is in graph_svd.jl.
# Fermionic sectors need no Jordan-Wigner strings: the odd-parity bond charge carries the braiding.
#
# Invariant to keep in mind: a started term whose running charge fused back to `unit` is
# indistinguishable from a pending one over its idle sites, so their suffix classes can collide
# (see `_promote_pending!`).

using TensorKit: Sector, Vect, block, sectors, space, dim
using MatrixAlgebraKit: svd_trunc, trunctol
using SparseArrays: SparseMatrixCSC

# The strategies with a `_bond_basis!` method (`IndependentSVD` has its own sweep).
const GraphStrategy = Union{VertexCover, SequentialSVD}

# A left (prefix) vertex: it entered the current site on bond index `link` and applies `key` here.
struct LeftVertex{I <: Sector}
    link::Int
    key::Factor{I}
end

# The persistent bipartite graph over an `TermTable`, handed from one site step to the next. A right
# vertex is a suffix class: classes enter at their term's first active site and thereafter only merge.
# `lefts` ↔ right vertices, with per-left adjacency `radj`/`wadj` (right id, weight; zeros never kept).
# Geometric (`ExpChannel`) classes: `rchan[r] > 0` is an automaton state, `-1` the synthetic exhausted
# class a channel exits onto. `startleft`/`startidx` (sentinel's left vertex and the start channel's
# outgoing index) and `rfinish`/`finishleft`/`finishidx` (the mirror) are 0 when absent.
mutable struct SweepGraph{I <: Sector}
    tt::TermTable{I}
    N::Int
    K::Int                 # arity(tt): rows of tt.sites / tt.factors
    sufid::Matrix{Int}     # (K+1) × M interned id of the column suffix j:K (0 == exhausted)
    relsufid::Matrix{Int}  # (K+1) × M same, interned by shape instead of position (`_rel_suffix_ids`)
    channels::Vector{ExpChannel{I}}   # lowered exponentially decaying interactions (may be empty)
    lastentry::Int         # last site at which a channel may still enter (0 if there are none)
    rrepr::Vector{Int}     # right vertex -> representative term id (0 for a geometric class)
    rcur::Vector{Int}      # right vertex -> first column j with sites[j, rrepr] > current site
    rbond::Vector{I}       # right vertex -> running bond charge just past the current site
    rchan::Vector{Int}     # right vertex -> 0 term class / >0 channel index / -1 exhausted class
    rstate::Vector{Int}    # right vertex -> automaton state within `channels[rchan]` (0 otherwise)
    lefts::Vector{LeftVertex{I}}
    radj::Vector{Vector{Int}}
    wadj::Vector{Vector{ComplexF64}}
    nlinks::Int
    lazy::Bool             # insert a term's right vertex only once it is reachable
    firstsite::Vector{Int} # term -> first active site (1 for a K=0 identity term)
    pend_at::Vector{Vector{Int}}                # site -> terms whose first active site is that site
    pendbysig::Dictionary{Tuple{Int, I}, Int}   # pre-start suffix signature -> pending term
    inserted::BitVector    # term -> already represented by a right vertex
    nremaining::Int        # terms not yet inserted (all of them start strictly right of here)
    rsent::Int             # sentinel right-vertex id for this bond (0 if none)
    startleft::Int         # left vertex carrying the sentinel (0 if none)
    startidx::Int          # outgoing bond index of the start channel (0 if there is none)
    rfinish::Int           # right vertex of the exhausted/trivial-charge class (0 if none)
    finishleft::Int        # left vertex that class's pass-through enters on (0 if none)
    finishidx::Int         # outgoing bond index of the finish channel (0 if there is none)
    jordan::Bool           # force the finish class into the cover (see `_force_finish!`)
    slot::Vector{Int}      # scratch: right-vertex id -> position within one left vertex's adjacency
    vlocal::Vector{Int}    # scratch: right-vertex id -> component-local index
    firstleft::Vector{Int} # right-vertex id -> first incident left vertex (0 if isolated)
    remap::Vector{Int}     # scratch: old right-vertex id -> merged right-vertex id
    siggroups::Dictionary{Tuple{Int, I}, Int}  # scratch: suffix signature -> merged right-vertex id
end

# Canonical translation-invariant name `(kind, distance, shape, charge)` of right vertex `r` at bond
# `i`: translates of each other share it. Kind 1 is a geometric class (named by its state); the
# synthetic exhausted class shares the exhausted term class's name; the sentinel is `(-1, -1, -1, 1)`.
function _rdesc(g::SweepGraph{I}, r::Int, i::Int) where {I}
    r > length(g.rrepr) && return (-1, -1, -1, unit(I))    # the sentinel
    c = g.rchan[r]
    c == -1 && return (0, 0, 0, g.rbond[r])                # synthetic exhausted class
    c > 0 && return (1, 0, g.channels[c].states[g.rstate[r]].name, g.rbond[r])
    t = g.rrepr[r]
    j = g.rcur[r]
    s = j <= g.K ? g.tt.sites[j, t] : 0
    return (0, iszero(s) ? 0 : s - i, g.relsufid[j, t], g.rbond[r])
end

# Advance `r`'s cursor past every factor at a site `<= i` (amortised `O(1)`) and return its suffix
# signature `(factor-list id, running bond charge)`. A geometric class takes a negative id (disjoint
# from `sufid`); the synthetic exhausted class takes `(0, charge)`, merging with exhausted term classes.
function _signature!(g::SweepGraph{I}, r::Int, i::Int) where {I}
    c = g.rchan[r]
    c == -1 && return (0, g.rbond[r])
    c > 0 && return (-g.channels[c].states[g.rstate[r]].name, g.rbond[r])
    t = g.rrepr[r]
    sites, keys = g.tt.sites, g.tt.factors
    j = g.rcur[r]
    @inbounds while j <= g.K
        s = sites[j, t]
        (iszero(s) || s > i) && break
        g.rbond[r] = keys[j, t].bond
        j += 1
    end
    g.rcur[r] = j
    return (g.sufid[j, t], g.rbond[r])
end

# Is this a *cyclic* channel state (forced into the cover, see `_forced_cover`)? channel state (live at every bond, forced into the cover, see `_forced_cover`)?
function _iscyclic(g::SweepGraph, r::Int)
    r <= length(g.rchan) || return false          # the sentinel
    c = g.rchan[r]
    return c > 0 && g.channels[c].states[g.rstate[r]].cyclic
end

# No remaining factors (term classes with a spent cursor and the synthetic exit class).
function _isexhausted(g::SweepGraph, r::Int)
    g.rchan[r] == -1 && return true
    return iszero(g.rchan[r]) && iszero(g.sufid[g.rcur[r], g.rrepr[r]])
end

# Append a right vertex (term class, geometric `(chan, state)`, or `chan == -1` exhausted) and return its id.
function _push_right!(arrs::Tuple, repr::Int, cur::Int, bond, chan::Int, state::Int)
    rrepr, rcur, rbond, rchan, rstate = arrs
    push!(rrepr, repr); push!(rcur, cur); push!(rbond, bond); push!(rchan, chan); push!(rstate, state)
    return length(rrepr)
end
_push_right!(g::SweepGraph, args...) = _push_right!((g.rrepr, g.rcur, g.rbond, g.rchan, g.rstate), args...)

# The left vertex for `(link, key)`, created on first use.
function _bucket!(buckets, lefts, radj, wadj, link::Int, key)
    b = get(buckets, (link, key), 0)
    if iszero(b)
        push!(lefts, LeftVertex(link, key))
        push!(radj, Int[])
        push!(wadj, ComplexF64[])
        b = length(lefts)
        insert!(buckets, (link, key), b)
    end
    return b
end

# Seed the graph in `Θ(M·K)`. `lazy` gives a right vertex only to terms active at site 1, the rest being
# represented by a sentinel and inserted when reachable (`Θ(Σ span)` instead of `Θ(N·M)`); `channels`
# (lowered `expterm`s) need `lazy`. `jordan` forces the finish class into every cover.
function SweepGraph(
        tt::TermTable{I}, N::Int; lazy::Bool = true, jordan::Bool = false,
        channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    M = nterms(tt)
    (lazy || isempty(channels)) ||
        throw(ArgumentError("exponentially decaying channels need the lazy sweep"))
    sufid = _suffix_ids(tt)
    relsufid = _rel_suffix_ids(tt)
    # (1 for a `K=0` identity term)
    firstsite = Int[max(1, tt.sites[1, t]) for t in 1:nterms(tt)]
    lastent = maximum(c -> lastentry(c, N), channels; init = 0)

    rrepr, rcur, rchan, rstate = Int[], Int[], Int[], Int[]
    rbond = I[]
    inserted = falses(M)
    lefts = LeftVertex{I}[]
    radj = Vector{Int}[]
    wadj = Vector{ComplexF64}[]
    buckets = Dictionary{Tuple{Int, Factor{I}}, Int}()
    g0 = (rrepr, rcur, rbond, rchan, rstate)

    for t in 1:M
        (lazy && firstsite[t] > 1) && continue
        b = _bucket!(buckets, lefts, radj, wadj, 1, _factor_at(tt, t, 1))
        inserted[t] = true
        push!(radj[b], _push_right!(g0, t, 1, unit(I), 0, 0))
        push!(wadj[b], tt.coeffs[t])
    end

    # channels anchored at site 1 enter on the same start channel a term does
    for (ci, c) in enumerate(channels)
        isentry(c, 1, N) || continue
        b = _bucket!(buckets, lefts, radj, wadj, 1, c.entrykey)
        push!(radj[b], _push_right!(g0, 0, arity(tt) + 1, c.states[c.start].bond, ci, c.start))
        push!(wadj[b], c.coeff)
    end

    # pending terms, keyed by pre-start signature (full factor list, trivial charge) for `_promote_pending!`
    pend_at = [Int[] for _ in 1:N]
    pendbysig = Dictionary{Tuple{Int, I}, Int}()
    nremaining = 0
    for t in 1:M
        inserted[t] && continue
        nremaining += 1
        push!(pend_at[firstsite[t]], t)
        insert!(pendbysig, (sufid[1, t], unit(I)), t)   # injective: `sufid[1, t]` fixes the term
    end

    # the start channel's left vertex, which the sentinel and every injected term or entry hangs off
    startleft = (nremaining > 0 || 1 < lastent) ?
        _bucket!(buckets, lefts, radj, wadj, 1, Factor{I}(passthrough(I), unit(I))) : 0

    cap = length(rrepr) + 1   # right-vertex ids are at most one sentinel beyond the live ones
    return SweepGraph{I}(
        tt, N, arity(tt), sufid, relsufid, channels, lastent,
        rrepr, rcur, rbond, rchan, rstate, lefts, radj, wadj, 1,
        lazy, firstsite, pend_at, pendbysig, inserted, nremaining, 0, startleft, 0, 0, 0, 0, jordan,
        zeros(Int, cap), zeros(Int, cap), zeros(Int, cap), zeros(Int, cap),
        Dictionary{Tuple{Int, I}, Int}()
    )
end

# Phase 1: merge right vertices with equal `_signature!`; returns `remap[old] -> new` and replaces `g.r*`.
function _suffix_merge!(g::SweepGraph{I}, i::Int) where {I}
    R = length(g.rrepr)
    groups = g.siggroups
    empty!(groups)
    remap = g.remap
    length(remap) < R && resize!(remap, R)

    new = (Int[], Int[], I[], Int[], Int[])
    for r in 1:R
        sig = _signature!(g, r, i)
        b = get(groups, sig, 0)
        if iszero(b)
            b = _push_right!(new, g.rrepr[r], g.rcur[r], g.rbond[r], g.rchan[r], g.rstate[r])
            insert!(groups, sig, b)
        end
        remap[r] = b
    end
    g.rrepr, g.rcur, g.rbond, g.rchan, g.rstate = new
    return remap
end

# Apply a right-vertex remap (or `nothing`) to `lv`'s adjacency, summing parallel edges and dropping
# cancelled ones (they must not reach the cover). In place via `slot`: `Θ(deg)`, no allocation.
function _merge_edges!(g::SweepGraph, lv::Int, remap)
    slot = g.slot
    radj, wadj = g.radj[lv], g.wadj[lv]
    n = 0
    @inbounds for k in eachindex(radj)
        rid = remap === nothing ? radj[k] : remap[radj[k]]
        s = slot[rid]
        if iszero(s)
            n += 1                  # n <= k always, so writing back into the same vectors is safe
            radj[n] = rid
            wadj[n] = wadj[k]
            slot[rid] = n
        else
            wadj[s] += wadj[k]
        end
    end
    @inbounds for k in 1:n
        slot[radj[k]] = 0           # reset only the touched slots
    end
    m = 0
    @inbounds for k in 1:n
        iszero(wadj[k]) && continue
        m += 1
        radj[m] = radj[k]
        wadj[m] = wadj[k]
    end
    resize!(radj, m)
    resize!(wadj, m)
    return m
end

# The scratch vectors are all-zero between uses, so only new space needs clearing.
function _grow_scratch!(g::SweepGraph, nV::Int)
    for scratch in (g.slot, g.vlocal, g.firstleft)
        if length(scratch) < nV
            n0 = length(scratch)
            resize!(scratch, nV)
            fill!(view(scratch, (n0 + 1):nV), 0)
        end
    end
    return g
end

# Insert every not-yet-started term whose suffix class *coincides* with a live one at bond `i`.
# `_factor_at` pads idle sites with a pass-through at the running charge, so a started term whose charge
# fused back to `unit` is indistinguishable from a pending one; the eager sweep would merge their right
# vertices, and this reproduces that at the earliest bond where it applies (one probe per live vertex).
function _promote_pending!(g::SweepGraph{I}, i::Int) where {I}
    (g.nremaining > 0 && !isempty(g.pendbysig)) || return g
    lv = g.startleft
    iszero(lv) && invariant_violation("pending terms with no start channel to inject them on")
    promoted = false
    for r in eachindex(g.rrepr)
        iszero(g.rchan[r]) || continue     # geometric classes have no factor list to collide with
        t = get(g.pendbysig, (g.sufid[g.rcur[r], g.rrepr[r]], g.rbond[r]), 0)
        (iszero(t) || g.inserted[t]) && continue
        g.inserted[t] = true
        g.nremaining -= 1
        push!(g.radj[lv], r)
        push!(g.wadj[lv], g.tt.coeffs[t])
        promoted = true
    end
    # the start channel may already have an edge to this class, so sum parallel edges
    promoted && _merge_edges!(g, lv, nothing)
    return g
end

# Renumber the right vertices into ascending `_rdesc` order and sort every adjacency list, so the sweep
# depends only on the canonical graph, not its construction history (Hopcroft–Karp and König depend on
# scan order). Needed on a periodic lattice, where the bond basis must converge exactly, not up to
# permutation. Called before the sentinel is attached.
function _canonicalise_rights!(g::SweepGraph{I}, i::Int) where {I}
    R = length(g.rrepr)
    if R > 1
        descs = [_rdesc(g, r, i) for r in 1:R]
        if !issorted(descs)
            perm = sortperm(descs)
            newid = Vector{Int}(undef, R)
            for p in 1:R
                newid[perm[p]] = p
            end
            g.rrepr, g.rcur, g.rbond = g.rrepr[perm], g.rcur[perm], g.rbond[perm]
            g.rchan, g.rstate = g.rchan[perm], g.rstate[perm]
            for radj in g.radj
                @inbounds for k in eachindex(radj)
                    radj[k] = newid[radj[k]]
                end
            end
        end
    end
    for iu in eachindex(g.radj)
        radj, wadj = g.radj[iu], g.wadj[iu]
        if !issorted(radj)
            p = sortperm(radj)
            permute!(radj, p)
            permute!(wadj, p)
        end
    end
    return g
end

# Phases 1 & 2 (shared by the strategies): suffix-merge, promote colliding pending terms, canonicalise,
# attach the sentinel for the still-pending ones (id one past the real right vertices, discarded each
# bond), and record `g.firstleft` while checking charge purity. Returns `(nU, nV)`.
function _prepare_bond!(g::SweepGraph{I}, i::Int) where {I}
    g.rsent = 0
    remap = _suffix_merge!(g, i)
    nV = length(g.rrepr)
    _grow_scratch!(g, nV)
    for lv in eachindex(g.lefts)
        _merge_edges!(g, lv, remap)
    end
    _promote_pending!(g, i)
    _canonicalise_rights!(g, i)

    # the finish class: exhausted at the trivial charge (unique after the merge)
    g.rfinish = something(findfirst(r -> _isexhausted(g, r) && g.rbond[r] == unit(I), eachindex(g.rrepr)), 0)

    if g.nremaining > 0 || i < g.lastentry
        nV += 1
        g.rsent = nV
        _grow_scratch!(g, nV)
        push!(g.radj[g.startleft], g.rsent)
        push!(g.wadj[g.startleft], one(ComplexF64))
    end

    nU = length(g.lefts)
    firstleft = g.firstleft
    fill!(view(firstleft, 1:nV), 0)
    for iu in 1:nU
        bond = g.lefts[iu].key.bond
        for rid in g.radj[iu]
            if iszero(firstleft[rid])
                firstleft[rid] = iu
            else
                g.lefts[firstleft[rid]].key.bond == bond ||
                    invariant_violation("bond index not sector-pure (block-diagonality violated)")
            end
        end
    end

    return nU, nV
end

# Phase 5: turn `g` into the graph for bond `i → i+1` on the same right vertices, with fresh left
# vertices linked by outgoing index `j` and keyed by the right vertex's next-site key.
# `nextedges_global[j]` = the `(right vertex, weight)` edges index `j` forwards.
# Lazy insertion injects here: terms and channel entries starting at `i+1` hang off the start channel.
# A geometric state has one successor per automaton transition (continue with weight `λ·c`, or exit);
# transitions that cannot complete inside `1:N` are dropped.
function _build_next_graph!(
        g::SweepGraph{I}, i::Int, nout::Int,
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}}
    ) where {I}
    # the key each term class applies at `i+1` (`_factor_at` in `O(1)` from the cursor); channel
    # states are handled below
    nextkeys = map(eachindex(g.rrepr)) do r
        t = g.rrepr[r]
        j = g.rcur[r]
        (j <= g.K && g.tt.sites[j, t] == i + 1) && return g.tt.factors[j, t]
        return Factor{I}(passthrough(I), g.rbond[r])
    end
    buckets = Dictionary{Tuple{Int, Factor{I}}, Int}()
    next_lefts = LeftVertex{I}[]
    next_radj = Vector{Int}[]
    next_wadj = Vector{ComplexF64}[]
    bucket!(j, key) = _bucket!(buckets, next_lefts, next_radj, next_wadj, j, key)

    # automaton successor states and the exit class are created on demand
    geom = Dictionary{Tuple{Int, Int}, Int}()
    exhausted = Dictionary{I, Int}()
    for r in eachindex(g.rrepr)
        if g.rchan[r] > 0
            insert!(geom, (g.rchan[r], g.rstate[r]), r)
        elseif g.rchan[r] == -1 || _isexhausted(g, r)
            haskey(exhausted, g.rbond[r]) || insert!(exhausted, g.rbond[r], r)
        end
    end
    function geomvertex!(chan::Int, state::Int, bond::I)
        iszero(state) && return get!(() -> _push_right!(g, 0, g.K + 1, bond, -1, 0), exhausted, bond)
        return get!(() -> _push_right!(g, 0, g.K + 1, bond, chan, state), geom, (chan, state))
    end

    for j in 1:nout
        for (rid, w) in nextedges_global[j]
            chan = g.rchan[rid]
            if chan > 0
                c = g.channels[chan]
                for (key, target, tw) in c.states[g.rstate[rid]].trans
                    bond = iszero(target) ? key.bond : c.states[target].bond
                    minremain = iszero(target) ? 0 : c.states[target].minremain
                    i + 1 + minremain <= g.N || continue
                    b = bucket!(j, key)
                    push!(next_radj[b], geomvertex!(chan, target, bond))
                    push!(next_wadj[b], w * tw)
                end
            else
                b = bucket!(j, nextkeys[rid])
                push!(next_radj[b], rid)
                push!(next_wadj[b], w)
            end
        end
    end

    if g.lazy
        startidx = g.startidx
        ((iszero(g.nremaining) && i + 1 > g.lastentry) || !iszero(startidx)) ||
            invariant_violation("terms remain to the right of site $i with no start channel to enter on")
        for t in g.pend_at[i + 1]
            g.inserted[t] && continue         # already promoted into a colliding class
            g.inserted[t] = true
            g.nremaining -= 1
            b = bucket!(startidx, g.tt.factors[1, t])
            push!(next_radj[b], _push_right!(g, t, 1, unit(I), 0, 0))
            push!(next_wadj[b], g.tt.coeffs[t])
        end
        # channels whose phase matches `i+1` enter on the start channel
        for (ci, c) in enumerate(g.channels)
            isentry(c, i + 1, g.N) || continue
            b = bucket!(startidx, c.entrykey)
            st = c.states[c.start]
            push!(next_radj[b], geomvertex!(ci, c.start, st.bond))
            push!(next_wadj[b], c.coeff)
        end
        # the start channel must survive even when it forwards nothing (the next sentinel hangs off it)
        g.startleft = (g.nremaining > 0 || i + 1 < g.lastentry) ?
            bucket!(startidx, Factor{I}(passthrough(I), unit(I))) : 0
    end

    # the finish channel forwards one edge, to the exhausted class, via the trivial pass-through
    g.finishleft = 0
    if !iszero(g.finishidx)
        g.finishleft = get(buckets, (g.finishidx, Factor{I}(passthrough(I), unit(I))), 0)
        iszero(g.finishleft) &&
            invariant_violation("the finish channel at bond $i forwards nothing to site $(i + 1)")
    end

    g.lefts = next_lefts
    g.radj = next_radj
    g.wadj = next_wadj
    g.nlinks = nout
    return g
end

# One site step: `_prepare_bond!`, the strategy's `_bond_basis!`, `_build_next_graph!`. Returns
# `(W_i, bondsectors_i, startidx, finishidx)`.
function _at_site!(g::SweepGraph{I}, i::Int, strategy::GraphStrategy = VertexCover()) where {I}
    nU, nV = _prepare_bond!(g, i)
    g.startidx = 0
    g.finishidx = 0
    nout, site_dict, secW, nextedges_global = _bond_basis!(g, i, nU, nV, strategy)
    Ws_i = sparse_from_dict(site_dict, (g.nlinks, nout))
    i < g.N && _build_next_graph!(g, i, nout, nextedges_global)
    return Ws_i, secW, g.startidx, g.finishidx
end

# The sparse adjacency makes lazy insertion pay off; the SVD builds a dense matrix anyway.
_graph_lazy(::VertexCover) = true
_graph_lazy(::SequentialSVD) = false

_emptysweep(::Type{I}) where {I} = (SparseMatrixCSC{SiteOperator{I}, Int}[], Vector{I}[], Int[], Int[])

# The graph sweep: `(Ws, bondsectors, starts, finishes)`, `starts[i]`/`finishes[i]` being the bond indices
# of the start/finish identity channels right of site `i` (0 where the cover spent none). `jordan` forces
# the finish class into the cover (which can cost one index, e.g. a constant next to two-site terms).
function _sweep_channels(
        tt::TermTable{I}, N::Int, strategy::GraphStrategy;
        jordan::Bool = true, channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    (nterms(tt) == 0 && isempty(channels)) && return _emptysweep(I)
    g = SweepGraph(tt, N; lazy = _graph_lazy(strategy), jordan, channels)
    Ws = Vector{SparseMatrixCSC{SiteOperator{I}, Int}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    starts = zeros(Int, N)
    finishes = zeros(Int, N)
    for i in 1:N
        Ws[i], bondsectors[i], starts[i], finishes[i] = _at_site!(g, i, strategy)
    end
    return (Ws, bondsectors, starts, finishes)
end

# `_sweep_channels` without forcing the finish class or reporting the identity channels.
function _sweep(
        tt::TermTable{I}, N::Int, strategy::GraphStrategy; channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    return _sweep_channels(tt, N, strategy; jordan = false, channels)[1:2]
end
