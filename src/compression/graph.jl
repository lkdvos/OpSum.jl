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

# The strategies with a `bond_basis!` method (`IndependentSVD` has its own sweep).
const GraphStrategy = Union{VertexCover, SequentialSVD}

# What a right vertex is: a suffix class of the terms, a state of a lowered `ExpChannel`'s automaton (its
# channel index in `rchan`), or the synthetic exhausted class a channel exits onto.
@enum RightKind TermClass AutomatonState ExhaustedClass

# The kind of a vertex in its canonical name (`rdesc`). The declaration order is the sort order of the names:
# sentinel, then term classes (the exhausted class included), then automaton states.
@enum DescKind SentinelDesc TermDesc AutomatonDesc

# A left (prefix) vertex: it entered the current site on bond index `link` and applies `key` here.
struct LeftVertex{I <: Sector}
    link::Int
    key::Factor{I}
end

# The right vertices, one column per property. An entry is a suffix class, an automaton state of a lowered
# `ExpChannel`, or the synthetic exhausted class (see `RightKind`). The columns only grow or are replaced
# as a whole (`SweepGraph.right`), never resized in place from outside.
Base.@kwdef struct RightVertices{I <: Sector}
    repr::Vector{Int} = Int[]        # representative term id (0 for an automaton state)
    cur::Vector{Int} = Int[]         # first column j with sites[j, repr] > current site
    bond::Vector{I} = I[]            # running bond charge just past the current site
    kind::Vector{RightKind} = RightKind[]
    chan::Vector{Int} = Int[]        # channel index (0 unless an automaton state)
    state::Vector{Int} = Int[]       # automaton state within `channels[chan]` (0 otherwise)
end
Base.length(r::RightVertices) = length(r.repr)

# The left vertices of a bond with their adjacency: `radj[u]`/`wadj[u]` list the right ids and weights vertex
# `u` is joined to (zeros never kept). `nlinks` is the size of the incoming bond they enter on.
Base.@kwdef struct LeftVertices{I <: Sector}
    vertex::Vector{LeftVertex{I}} = LeftVertex{I}[]
    radj::Vector{Vector{Int}} = Vector{Int}[]
    wadj::Vector{Vector{ComplexF64}} = Vector{ComplexF64}[]
    nlinks::Int
end

# Lazy insertion: a term gets a right vertex only once it is reachable, so until then it is `pending`.
Base.@kwdef mutable struct Pending{I <: Sector}
    firstsite::Vector{Int}                      # term -> first active site (1 for a K=0 identity term)
    at::Vector{Vector{Int}}                     # site -> terms whose first active site is that site
    bysig::Dictionary{Tuple{Int, I}, Int}       # pre-start suffix signature -> pending term
    inserted::BitVector                         # term -> already represented by a right vertex
    remaining::Int                              # terms not yet inserted (all start strictly right of here)
end

# Where the identity channels sit in the current bond; 0 when absent. `startleft`/`startidx`: the left
# vertex carrying the sentinel and the start channel's outgoing index. `rfinish`/`finishleft`/`finishidx`
# are the mirror (the exhausted, trivial-charge class, the left vertex its pass-through enters on, and the
# finish channel's outgoing index). `rsent` is the sentinel right vertex's id.
Base.@kwdef mutable struct Boundary
    rsent::Int = 0
    startleft::Int = 0
    startidx::Int = 0
    rfinish::Int = 0
    finishleft::Int = 0
    finishidx::Int = 0
end

# Per-bond scratch, all-zero between uses (see `_grow_scratch!`) except `siggroups`, emptied per use.
Base.@kwdef struct Scratch{I <: Sector}
    slot::Vector{Int}          # right-vertex id -> position within one left vertex's adjacency
    vlocal::Vector{Int}        # right-vertex id -> component-local index
    firstleft::Vector{Int}     # right-vertex id -> first incident left vertex (0 if isolated)
    remap::Vector{Int}         # old right-vertex id -> merged right-vertex id
    siggroups::Dictionary{Tuple{Int, I}, Int} = Dictionary{Tuple{Int, I}, Int}()  # suffix signature -> merged id
end

# The persistent bipartite graph over a `TermTable`, handed from one site step to the next. A right
# vertex is a suffix class: classes enter at their term's first active site and thereafter only merge.
# `right` and `left` are replaced wholesale each bond; `pending` and `bound` are updated in place.
Base.@kwdef mutable struct SweepGraph{I <: Sector}
    tt::TermTable{I}
    N::Int
    sufid::Matrix{Int}     # (K+1) × M interned id of the column suffix j:K (0 == exhausted)
    relsufid::Matrix{Int}  # (K+1) × M same, interned by shape instead of position (`rel_suffix_ids`)
    channels::Vector{ExpChannel{I}}   # lowered exponentially decaying interactions (may be empty)
    lastentry::Int         # last site at which a channel may still enter (0 if there are none)
    lazy::Bool             # insert a term's right vertex only once it is reachable (else all up front)
    jordan::Bool           # force the finish class into the cover (see `_force_finish!`)
    right::RightVertices{I}
    left::LeftVertices{I}
    pending::Pending{I}
    bound::Boundary
    scratch::Scratch{I}
end

# Canonical translation-invariant name `(kind, distance, shape, charge)` of right vertex `r` at bond
# `i`: translates of each other share it. An automaton state is named by its state; the synthetic
# exhausted class shares the exhausted term class's name; the sentinel is `(SentinelDesc, 0, 0, unit)`.
function rdesc(g::SweepGraph{I}, r::Int, i::Int) where {I}
    r > length(g.right.repr) && return (SentinelDesc, 0, 0, unit(I))
    kind = g.right.kind[r]
    kind == ExhaustedClass && return (TermDesc, 0, 0, g.right.bond[r])     # synthetic exhausted class
    kind == AutomatonState && return (AutomatonDesc, 0, g.channels[g.right.chan[r]].states[g.right.state[r]].name, g.right.bond[r])
    t = g.right.repr[r]
    j = g.right.cur[r]
    s = j <= arity(g.tt) ? g.tt.sites[j, t] : 0
    return (TermDesc, iszero(s) ? 0 : s - i, g.relsufid[j, t], g.right.bond[r])
end

# Advance `r`'s cursor past every factor at a site `<= i` (amortised `O(1)`) and return its suffix
# signature `(factor-list id, running bond charge)`. An automaton state takes a negative id (disjoint
# from `sufid`); the synthetic exhausted class takes `(0, charge)`, merging with exhausted term classes.
function _signature!(g::SweepGraph{I}, r::Int, i::Int) where {I}
    kind = g.right.kind[r]
    kind == ExhaustedClass && return (0, g.right.bond[r])
    kind == AutomatonState && return (-g.channels[g.right.chan[r]].states[g.right.state[r]].name, g.right.bond[r])
    t = g.right.repr[r]
    sites, keys = g.tt.sites, g.tt.factors
    j = g.right.cur[r]
    @inbounds while j <= arity(g.tt)
        s = sites[j, t]
        (iszero(s) || s > i) && break
        g.right.bond[r] = keys[j, t].bond
        j += 1
    end
    g.right.cur[r] = j
    return (g.sufid[j, t], g.right.bond[r])
end

# Is this a *cyclic* channel state (live at every bond, so forced into the cover, see `_forced_cover`)?
function iscyclic(g::SweepGraph, r::Int)
    r <= length(g.right.kind) || return false          # the sentinel
    g.right.kind[r] == AutomatonState || return false
    return g.channels[g.right.chan[r]].states[g.right.state[r]].cyclic
end

# No remaining factors (term classes with a spent cursor and the synthetic exit class).
function _isexhausted(g::SweepGraph, r::Int)
    g.right.kind[r] == ExhaustedClass && return true
    return g.right.kind[r] == TermClass && iszero(g.sufid[g.right.cur[r], g.right.repr[r]])
end

# Append a right vertex (a term class, an automaton `(chan, state)`, or the exhausted class) and return its id.
function _push_right!(r::RightVertices, repr::Int, cur::Int, bond, kind::RightKind, chan::Int, state::Int)
    push!(r.repr, repr); push!(r.cur, cur); push!(r.bond, bond)
    push!(r.kind, kind); push!(r.chan, chan); push!(r.state, state)
    return length(r)
end

# The left vertex for `(link, key)`, created on first use.
function _bucket!(buckets, left::LeftVertices, link::Int, key)
    b = get(buckets, (link, key), 0)
    if iszero(b)
        push!(left.vertex, LeftVertex(link, key))
        push!(left.radj, Int[])
        push!(left.wadj, ComplexF64[])
        b = length(left.vertex)
        insert!(buckets, (link, key), b)
    end
    return b
end

# Seed the graph in `Θ(M·K)`: only terms active at site 1 get a right vertex, the rest being represented by
# a sentinel and inserted when reachable (`Θ(Σ span)` instead of `Θ(N·M)`) [lazy mode]. `channels` are
# lowered `expterm`s; `jordan` forces the finish class into every cover. `lazy = false` seeds every term
# up front, which `SequentialSVD` needs (see `_graph_lazy`).
function SweepGraph(
        tt::TermTable{I}, N::Int; lazy::Bool = true, jordan::Bool = false,
        channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    M = nterms(tt)
    (lazy || isempty(channels)) ||
        throw(ArgumentError("exponentially decaying channels need the lazy sweep"))
    sufid = suffix_ids(tt)
    relsufid = rel_suffix_ids(tt)
    # (1 for a `K=0` identity term)
    firstsite = Int[max(1, tt.sites[1, t]) for t in 1:nterms(tt)]
    lastent = maximum(c -> lastentry(c, N), channels; init = 0)

    right = RightVertices{I}()
    left = LeftVertices{I}(; nlinks = 1)
    inserted = falses(M)
    buckets = Dictionary{Tuple{Int, Factor{I}}, Int}()

    for t in 1:M
        (lazy && firstsite[t] > 1) && continue
        b = _bucket!(buckets, left, 1, factor_at(tt, t, 1))
        inserted[t] = true
        push!(left.radj[b], _push_right!(right, t, 1, unit(I), TermClass, 0, 0))
        push!(left.wadj[b], tt.coeffs[t])
    end

    # channels anchored at site 1 enter on the same start channel a term does
    for (ci, c) in enumerate(channels)
        isentry(c, 1, N) || continue
        b = _bucket!(buckets, left, 1, c.entrykey)
        push!(left.radj[b], _push_right!(right, 0, arity(tt) + 1, c.states[c.start].bond, AutomatonState, ci, c.start))
        push!(left.wadj[b], c.coeff)
    end

    # pending terms, keyed by pre-start signature (full factor list, trivial charge) for `_promote_pending!`
    pendingat = [Int[] for _ in 1:N]
    bysig = Dictionary{Tuple{Int, I}, Int}()
    remaining = 0
    for t in 1:M
        inserted[t] && continue
        remaining += 1
        push!(pendingat[firstsite[t]], t)
        insert!(bysig, (sufid[1, t], unit(I)), t)   # injective: `sufid[1, t]` fixes the term
    end

    # the start channel's left vertex, which the sentinel and every injected term or entry hangs off
    startleft = (remaining > 0 || 1 < lastent) ?
        _bucket!(buckets, left, 1, Factor{I}(passthrough(I), unit(I))) : 0

    cap = length(right) + 1   # right-vertex ids are at most one sentinel beyond the live ones
    return SweepGraph{I}(;
        tt, N, sufid, relsufid, channels, lastentry = lastent, lazy, jordan, right, left,
        pending = Pending{I}(; firstsite, at = pendingat, bysig, inserted, remaining),
        bound = Boundary(; startleft),
        scratch = Scratch{I}(; slot = zeros(Int, cap), vlocal = zeros(Int, cap), firstleft = zeros(Int, cap), remap = zeros(Int, cap))
    )
end

# Phase 1: merge right vertices with equal `_signature!`; returns `remap[old] -> new` and replaces `g.right`.
function _suffix_merge!(g::SweepGraph{I}, i::Int) where {I}
    R = length(g.right.repr)
    groups = g.scratch.siggroups
    empty!(groups)
    remap = g.scratch.remap
    length(remap) < R && resize!(remap, R)

    old = g.right
    new = RightVertices{I}()
    for r in 1:R
        sig = _signature!(g, r, i)
        b = get(groups, sig, 0)
        if iszero(b)
            b = _push_right!(new, old.repr[r], old.cur[r], old.bond[r], old.kind[r], old.chan[r], old.state[r])
            insert!(groups, sig, b)
        end
        remap[r] = b
    end
    g.right = new
    return remap
end

# Apply a right-vertex remap (or `nothing`) to `lv`'s adjacency, summing parallel edges and dropping
# cancelled ones (they must not reach the cover). In place via `slot`: `Θ(deg)`, no allocation.
function _merge_edges!(g::SweepGraph, lv::Int, remap)
    slot = g.scratch.slot
    radj, wadj = g.left.radj[lv], g.left.wadj[lv]
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
    for scratch in (g.scratch.slot, g.scratch.vlocal, g.scratch.firstleft)
        if length(scratch) < nV
            n0 = length(scratch)
            resize!(scratch, nV)
            fill!(view(scratch, (n0 + 1):nV), 0)
        end
    end
    return g
end

# Insert every not-yet-started term whose suffix class *coincides* with a live one at bond `i`.
# `factor_at` pads idle sites with a pass-through at the running charge, so a started term whose charge
# fused back to `unit` is indistinguishable from a pending one; the eager sweep would merge their right
# vertices, and this reproduces that at the earliest bond where it applies (one probe per live vertex).
function _promote_pending!(g::SweepGraph{I}, i::Int) where {I}
    (g.pending.remaining > 0 && !isempty(g.pending.bysig)) || return g
    lv = g.bound.startleft
    iszero(lv) && invariant_violation("pending terms with no start channel to inject them on")
    promoted = false
    for r in eachindex(g.right.repr)
        g.right.kind[r] == TermClass || continue     # the other kinds have no factor list to collide with
        t = get(g.pending.bysig, (g.sufid[g.right.cur[r], g.right.repr[r]], g.right.bond[r]), 0)
        (iszero(t) || g.pending.inserted[t]) && continue
        g.pending.inserted[t] = true
        g.pending.remaining -= 1
        push!(g.left.radj[lv], r)
        push!(g.left.wadj[lv], g.tt.coeffs[t])
        promoted = true
    end
    # the start channel may already have an edge to this class, so sum parallel edges
    promoted && _merge_edges!(g, lv, nothing)
    return g
end

# Renumber the right vertices into ascending `rdesc` order and sort every adjacency list, so the sweep
# depends only on the canonical graph, not its construction history (Hopcroft–Karp and König depend on
# scan order). Needed on a periodic lattice, where the bond basis must converge exactly, not up to
# permutation. Called before the sentinel is attached.
function _canonicalise_rights!(g::SweepGraph{I}, i::Int) where {I}
    R = length(g.right.repr)
    if R > 1
        descs = [rdesc(g, r, i) for r in 1:R]
        if !issorted(descs)
            perm = sortperm(descs)
            newid = Vector{Int}(undef, R)
            for p in 1:R
                newid[perm[p]] = p
            end
            old = g.right
            g.right = RightVertices{I}(;
                repr = old.repr[perm], cur = old.cur[perm], bond = old.bond[perm],
                kind = old.kind[perm], chan = old.chan[perm], state = old.state[perm]
            )
            for radj in g.left.radj
                @inbounds for k in eachindex(radj)
                    radj[k] = newid[radj[k]]
                end
            end
        end
    end
    for iu in eachindex(g.left.radj)
        radj, wadj = g.left.radj[iu], g.left.wadj[iu]
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
# bond), and record `g.scratch.firstleft` while checking charge purity. Returns `(nU, nV)`.
function _prepare_bond!(g::SweepGraph{I}, i::Int) where {I}
    g.bound.rsent = 0
    remap = _suffix_merge!(g, i)
    nV = length(g.right.repr)
    _grow_scratch!(g, nV)
    for lv in eachindex(g.left.vertex)
        _merge_edges!(g, lv, remap)
    end
    _promote_pending!(g, i)
    _canonicalise_rights!(g, i)

    # the finish class: exhausted at the trivial charge (unique after the merge)
    g.bound.rfinish = something(findfirst(r -> _isexhausted(g, r) && g.right.bond[r] == unit(I), eachindex(g.right.repr)), 0)

    if g.pending.remaining > 0 || i < g.lastentry
        nV += 1
        g.bound.rsent = nV
        _grow_scratch!(g, nV)
        push!(g.left.radj[g.bound.startleft], g.bound.rsent)
        push!(g.left.wadj[g.bound.startleft], one(ComplexF64))
    end

    nU = length(g.left.vertex)
    firstleft = g.scratch.firstleft
    fill!(view(firstleft, 1:nV), 0)
    for iu in 1:nU
        bond = g.left.vertex[iu].key.bond
        for rid in g.left.radj[iu]
            if iszero(firstleft[rid])
                firstleft[rid] = iu
            else
                g.left.vertex[firstleft[rid]].key.bond == bond ||
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
# An automaton state has one successor per automaton transition (continue with weight `λ·c`, or exit);
# transitions that cannot complete inside `1:N` are dropped.
function _build_next_graph!(
        g::SweepGraph{I}, i::Int, nout::Int,
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}}
    ) where {I}
    # the key each term class applies at `i+1` (`factor_at` in `O(1)` from the cursor); channel
    # states are handled below
    nextkeys = map(eachindex(g.right.repr)) do r
        t = g.right.repr[r]
        j = g.right.cur[r]
        (j <= arity(g.tt) && g.tt.sites[j, t] == i + 1) && return g.tt.factors[j, t]
        return Factor{I}(passthrough(I), g.right.bond[r])
    end
    buckets = Dictionary{Tuple{Int, Factor{I}}, Int}()
    next = LeftVertices{I}(; nlinks = nout)
    bucket!(j, key) = _bucket!(buckets, next, j, key)

    # automaton successor states and the exit class are created on demand
    geom = Dictionary{Tuple{Int, Int}, Int}()
    exhausted = Dictionary{I, Int}()
    for r in eachindex(g.right.repr)
        if g.right.kind[r] == AutomatonState
            insert!(geom, (g.right.chan[r], g.right.state[r]), r)
        elseif _isexhausted(g, r)
            haskey(exhausted, g.right.bond[r]) || insert!(exhausted, g.right.bond[r], r)
        end
    end
    function geomvertex!(chan::Int, state::Int, bond::I)
        iszero(state) && return get!(() -> _push_right!(g.right, 0, arity(g.tt) + 1, bond, ExhaustedClass, 0, 0), exhausted, bond)
        return get!(() -> _push_right!(g.right, 0, arity(g.tt) + 1, bond, AutomatonState, chan, state), geom, (chan, state))
    end

    for j in 1:nout
        for (rid, w) in nextedges_global[j]
            if g.right.kind[rid] == AutomatonState
                chan = g.right.chan[rid]
                c = g.channels[chan]
                for (key, target, tw) in c.states[g.right.state[rid]].trans
                    bond = iszero(target) ? key.bond : c.states[target].bond
                    minremain = iszero(target) ? 0 : c.states[target].minremain
                    i + 1 + minremain <= g.N || continue
                    b = bucket!(j, key)
                    push!(next.radj[b], geomvertex!(chan, target, bond))
                    push!(next.wadj[b], w * tw)
                end
            else
                b = bucket!(j, nextkeys[rid])
                push!(next.radj[b], rid)
                push!(next.wadj[b], w)
            end
        end
    end

    if g.lazy
        startidx = g.bound.startidx
        ((iszero(g.pending.remaining) && i + 1 > g.lastentry) || !iszero(startidx)) ||
            invariant_violation("terms remain to the right of site $i with no start channel to enter on")
        for t in g.pending.at[i + 1]
            g.pending.inserted[t] && continue         # already promoted into a colliding class
            g.pending.inserted[t] = true
            g.pending.remaining -= 1
            b = bucket!(startidx, g.tt.factors[1, t])
            push!(next.radj[b], _push_right!(g.right, t, 1, unit(I), TermClass, 0, 0))
            push!(next.wadj[b], g.tt.coeffs[t])
        end
        # channels whose phase matches `i+1` enter on the start channel
        for (ci, c) in enumerate(g.channels)
            isentry(c, i + 1, g.N) || continue
            b = bucket!(startidx, c.entrykey)
            st = c.states[c.start]
            push!(next.radj[b], geomvertex!(ci, c.start, st.bond))
            push!(next.wadj[b], c.coeff)
        end
        # the start channel must survive even when it forwards nothing (the next sentinel hangs off it)
        g.bound.startleft = (g.pending.remaining > 0 || i + 1 < g.lastentry) ?
            bucket!(startidx, Factor{I}(passthrough(I), unit(I))) : 0
    end

    # the finish channel forwards one edge, to the exhausted class, via the trivial pass-through
    g.bound.finishleft = 0
    if !iszero(g.bound.finishidx)
        g.bound.finishleft = get(buckets, (g.bound.finishidx, Factor{I}(passthrough(I), unit(I))), 0)
        iszero(g.bound.finishleft) &&
            invariant_violation("the finish channel at bond $i forwards nothing to site $(i + 1)")
    end

    g.left = next
    return g
end

# The compressed basis of one bond, the result of every strategy's `bond_basis!`: its `nout` indices with
# their charges (`sectors`), the entries `(incoming link, index) ↦ letter` of `W_i` (`site`), the
# `(right vertex, weight)` edges each index forwards (`nextedges`, empty at the last site) and the indices
# of the start/finish identity channels (0 when absent; only the vertex cover reserves them).
struct BondBasis{I <: Sector}
    nout::Int
    site::Dictionary{CartesianIndex{2}, SiteOperator{I}}
    sectors::Vector{I}
    nextedges::Vector{Vector{Tuple{Int, ComplexF64}}}
    startidx::Int
    finishidx::Int
end
BondBasis{I}(nout, site, sectors, nextedges) where {I} = BondBasis{I}(nout, site, sectors, nextedges, 0, 0)

# One site step: `_prepare_bond!`, the strategy's `bond_basis!`, `_build_next_graph!`. Returns
# `(W_i, bondsectors_i, startidx, finishidx)`.
function at_site!(g::SweepGraph{I}, i::Int, strategy::GraphStrategy = VertexCover()) where {I}
    nU, nV = _prepare_bond!(g, i)
    b = bond_basis!(g, i, nU, nV, strategy)
    g.bound.startidx = b.startidx
    g.bound.finishidx = b.finishidx
    Ws_i = sparse_from_dict(b.site, (g.left.nlinks, b.nout))
    i < g.N && _build_next_graph!(g, i, b.nout, b.nextedges)
    return Ws_i, b.sectors, b.startidx, b.finishidx
end

# Lazy insertion represents the not-yet-started terms by a sentinel right vertex and injects them later
# through the start channel's bond index (`startidx`). Only the vertex cover reserves such an index (the
# sentinel is never covered, so its left vertex is covered-left). `SequentialSVD` would take the sentinel
# for an ordinary column of its dense matrix, carry it forward as a vertex that does not exist, and have no
# start channel to inject on (tried: `BoundsError` in `_build_next_graph!`), so it seeds every term up front.
_graph_lazy(::VertexCover) = true
_graph_lazy(::SequentialSVD) = false

emptysweep(::Type{I}) where {I} = (; Ws = SparseMatrixCSC{SiteOperator{I}, Int}[], bondsectors = Vector{I}[], starts = Int[], finishes = Int[])

# The reduced-MPO sweep of a term table over `N` sites, as the named tuple `(; Ws, bondsectors, starts,
# finishes)`; `starts[i]`/`finishes[i]` are the bond indices of the start/finish identity channels right of
# site `i` (0 where the cover spent none). `jordan` forces the finish class into the cover (which can cost
# one index, e.g. a constant next to two-site terms); without it the cover is free to be one index smaller.
function sweep(
        tt::TermTable{I}, N::Int, strategy::GraphStrategy;
        jordan::Bool = true, channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    (nterms(tt) == 0 && isempty(channels)) && return emptysweep(I)
    g = SweepGraph(tt, N; lazy = _graph_lazy(strategy), jordan, channels)
    Ws = Vector{SparseMatrixCSC{SiteOperator{I}, Int}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    starts = zeros(Int, N)
    finishes = zeros(Int, N)
    for i in 1:N
        Ws[i], bondsectors[i], starts[i], finishes[i] = at_site!(g, i, strategy)
    end
    return (; Ws, bondsectors, starts, finishes)
end
