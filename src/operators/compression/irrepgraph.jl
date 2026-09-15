# Reduced-MPO sweeps for the ITO automaton. `VertexCover`/`SequentialSVD` run the persistent
# bipartite-graph sweep `_irrep_graph_sweep`; `IndependentSVD` compresses each bond independently via
# `_irrep_independent_svd`, sharing only the suffix/prefix interning. Cost is
# `Θ(M·K) + Θ(Σ_terms span)`, linear in `N` for finite-range models. See
# `research/persistent-graph-mpo.md` for the design (§2.2's pending/started collision is the
# invariant most likely to break here) and §1 for the non-abelian mapping.
#
# Fermionic (graded) sectors are supported without Jordan-Wigner strings: TensorKit's braiding
# carries the anticommutation through the ITO algebra via the odd-parity charge on the virtual bond.
#
# Invariant checks below are real `throw`s (via `@noinline _invariant`), never `@assert`, since a
# violation means silently wrong output, not a crash.

using TensorKit: Sector, Vect, block, sectors, space, dim
using MatrixAlgebraKit: svd_trunc, trunctol
using SparseArrays: SparseMatrixCSC

# Invariant violation. `@noinline` so the (cold) message construction never inlines into the hot
# loops that check these — the check itself is then one comparison and a never-taken branch, which is
# what lets every one of these be a real `throw` rather than a strippable `@assert`.
@noinline function _invariant(msg::AbstractString)
    return error("OpSum internal invariant violated: ", msg, ". This is a bug in OpSum; the reduced MPO would be wrong.")
end

# The bond-basis strategies the persistent-graph skeleton can run — the ones with a `_bond_basis!`
# method. `IndependentSVD` is deliberately not among them: see `_irrep_independent_svd`.
const GraphStrategy = Union{VertexCover, SequentialSVD}

"""
    LeftVertex{I}

A left (prefix) vertex of the persistent ITO graph: it entered the current site on incoming bond
index `link` and applies the on-site ITO key `key = (op, bond, vertex)` here. The non-abelian
analogue of ITensorMPOConstruction's `LeftVertex(link, op_id, needs_JW)`; the fermion/JW-string slot
is intentionally omitted, because in the symmetric setting the JW string is subsumed by the sector
structure (the odd-parity charge on the virtual bond does that bookkeeping).
"""
struct LeftVertex{I <: Sector}
    link::Int
    key::ITOKey{I}
end

"""
    ITOGraph{I}

The persistent bipartite graph over an [`ITOTermTable`](@ref), handed from one site step to the next
by `_at_site!`. A right vertex is a suffix class (identified by a representative term id); classes
enter at their term's first active site and thereafter only merge. The current bipartite graph (for the
bond `i-1 → i` about to be processed) is `lefts` ↔ right vertices, with per-left-vertex adjacency lists
`radj`/`wadj` (right-vertex id, scalar weight); zero-weight entries are never kept.

Fields split into three groups:
* fixed suffix-class machinery: `K` (the term table's arity) and `sufid`, the interned id of every
  *contiguous column suffix* of `tt` — see [`_suffix_ids`](@ref) — plus the lowered exponentially
  decaying `channels` and `lastentry`, the last site at which any of them may still enter;
* persistent right-vertex state (shrinks via the suffix-merge): `rrepr` (representative term id) plus
  the monotone cursor `rcur`/`rbond` that turns "the suffix path from site `i+1`" into an `O(1)`
  two-word signature, and `rchan`/`rstate`, which name the *geometric* classes instead: `rchan[r] == 0`
  is an ordinary term class, `rchan[r] > 0` an automaton state of that channel
  ([`ExpChannel`](@ref)), and `rchan[r] == -1` the synthetic exhausted class a channel exits onto;
* the current bipartite graph: `lefts`, `radj`, `wadj`, and `nlinks` (incoming bond dimension);
* lazy-insertion state (see [`_promote_pending!`](@ref)): `lazy`, `firstsite`, `pend_at`, `pendbysig`,
  `inserted`, `nremaining`, and the per-bond `rsent` (sentinel right-vertex id, 0 if none),
  `startleft` (the left vertex the sentinel hangs off, 0 if none) and `startidx` (the outgoing bond
  index of the start channel);
* finish-channel bookkeeping, the mirror image of the start channel and the other half of what
  Jordan emission needs (jordanmpo.jl): `rfinish` (the right vertex whose suffix class is exhausted
  at the trivial charge, 0 if there is none at this bond), `finishleft` (the left vertex that class's
  pass-through enters on, 0 if none) and `finishidx` (the outgoing bond index of the finish channel);
* per-site scratch reused across the sweep so the site step allocates nothing per bond: `slot` and
  `vlocal` (right-vertex id → local index, for the remap and the per-component numbering),
  `firstleft` (right vertex → first incident left vertex, all the cover needs to read off a covered
  right vertex's bond charge without transposing the adjacency), `remap`, and `siggroups`.
"""
mutable struct ITOGraph{I <: Sector}
    tt::ITOTermTable{I}
    N::Int
    K::Int                 # arity(tt): rows of tt.sites / tt.keys
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

"""
    _rdesc(g::ITOGraph, r, i) -> (kind, distance, shape, charge)

The canonical, translation-invariant name of right vertex `r` at bond `i`. For an ordinary term class
(`kind = 0`) it is the distance from the bond to its first remaining factor (`0` once the class is
exhausted), the shape of the remaining factor list ([`_rel_suffix_ids`](@ref)), and the running bond
charge. Two classes carry the same name exactly when they are translates of each other relative to
their bonds, which is what makes the sweep's choices comparable from one unit cell to the next.

A geometric class (`kind = 1`) is named by its channel-automaton state's interned name, which is
position-independent by construction ([`ChannelState`](@ref)); the synthetic exhausted class shares
`kind = 0`'s exhausted name so that it sorts — and merges — with exhausted term classes. The sentinel
is not a suffix class at all and gets the reserved name `(-1, -1, -1, unit(I))`, which sorts before
every real one.
"""
function _rdesc(g::ITOGraph{I}, r::Int, i::Int) where {I}
    r > length(g.rrepr) && return (-1, -1, -1, unit(I))    # the sentinel
    c = g.rchan[r]
    c == -1 && return (0, 0, 0, g.rbond[r])                # synthetic exhausted class
    c > 0 && return (1, 0, g.channels[c].states[g.rstate[r]].name, g.rbond[r])
    t = g.rrepr[r]
    j = g.rcur[r]
    s = j <= g.K ? g.tt.sites[j, t] : 0
    return (0, iszero(s) ? 0 : s - i, g.relsufid[j, t], g.rbond[r])
end

# Advance right vertex `r`'s cursor past every factor at a site `<= i`, accumulating the running bond
# charge, and return its suffix signature `(interned remaining-factor list, running bond charge)`.
# Amortised `O(1)`: each cursor advances at most `K` times over the whole sweep.
#
# A geometric class has no cursor: its automaton state already *is* the class, so its signature is the
# state's interned name in a disjoint (negative) id space — `sufid` ids are `>= 0`, so a channel state
# can never be confused with a term's remaining factor list, nor collide with `pendbysig`. The one
# deliberate exception is the synthetic exhausted class, which takes the ordinary `(0, charge)`
# signature so that it merges with exhausted term classes: that merge is the done channel.
function _signature!(g::ITOGraph{I}, r::Int, i::Int) where {I}
    c = g.rchan[r]
    c == -1 && return (0, g.rbond[r])
    c > 0 && return (-g.channels[c].states[g.rstate[r]].name, g.rbond[r])
    t = g.rrepr[r]
    sites, keys = g.tt.sites, g.tt.keys
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

# Is this right vertex a *cyclic* channel state — one that is live at every bond and has to be forced
# into the bond basis (see `_vc_component`)?
function _iscyclic(g::ITOGraph, r::Int)
    r <= length(g.rchan) || return false          # the sentinel
    c = g.rchan[r]
    return c > 0 && g.channels[c].states[g.rstate[r]].cyclic
end

# Has this right vertex no remaining factors at all? Both an ordinary term class whose cursor has run
# out and the synthetic class a channel exits onto are "exhausted": they are the same class (equal
# signatures), which is what makes a channel's exit land on the done channel.
function _isexhausted(g::ITOGraph, r::Int)
    g.rchan[r] == -1 && return true
    return iszero(g.rchan[r]) && iszero(g.sufid[g.rcur[r], g.rrepr[r]])
end

# Append a right vertex for the geometric class `(chan, state)` (or, with `chan == -1`, the synthetic
# exhausted class at charge `bond`) and return its id.
function _push_geometric!(g::ITOGraph{I}, chan::Int, state::Int, bond::I) where {I}
    push!(g.rrepr, 0)
    push!(g.rcur, g.K + 1)
    push!(g.rbond, bond)
    push!(g.rchan, chan)
    push!(g.rstate, state)
    return length(g.rrepr)
end

# The `ITOKey` term `r`'s class applies at site `i+1`, from the cursor state left by `_signature!(…, i)`
# — identical to `_op_at_ito(tt, rrepr[r], i+1)` but `O(1)` and without rebuilding the pass-through.
function _next_key(g::ITOGraph{I}, r::Int, i::Int) where {I}
    t = g.rrepr[r]
    j = g.rcur[r]
    (j <= g.K && g.tt.sites[j, t] == i + 1) && return g.tt.keys[j, t]
    return ITOKey{I}(passthrough(I), g.rbond[r], 1)
end

# First active site of each term; a `K=0` identity term is treated as starting at site 1 (it is all
# pass-through, so it belongs to the identity/start channel from the very first bond). `tt.sites`
# columns are ascending and zero-padded, so this is just the first row.
function _first_sites(tt::ITOTermTable)
    return Int[(s = tt.sites[1, t]; iszero(s) ? 1 : s) for t in 1:nterms(tt)]
end

"""
    ITOGraph(tt::ITOTermTable{I}, N; lazy = true, channels = ExpChannel{I}[]) -> ITOGraph{I}

Seed the persistent graph (mirrors ITensor's `MPOGraph(os)`): intern the column suffixes, create the
right vertices, and bucket the initial left vertices by the site-1 key against the single
left-boundary link. `Θ(M·K)`, with no length-`N` path and no sort.

With `lazy = true` (the default) only terms *active at site 1* get a right vertex; the rest are
represented collectively by a sentinel right vertex on the identity/start channel and are inserted
when they become reachable (`_promote_pending!` / the injection in `_build_next_graph!`). That is what
makes the sweep cost `Θ(Σ_terms span)` instead of `Θ(N·M)`. `lazy = false` seeds every term eagerly,
which is the form the `SequentialSVD` strategy uses (its per-bond dense SVD dominates anyway).

`jordan = true` additionally forces the finish class into every cover (`_force_finish!`), which is
what Jordan emission needs and what `irrep_mpo` deliberately does not do.

`channels` are lowered exponentially decaying interactions ([`_lower_channels`](@ref)). They are not
terms and never enter the table: each *enters* on the start channel at every site matching its phase
(here for site 1, in `_build_next_graph!` for the rest), and thereafter is an ordinary right vertex
whose class happens to be cyclic. They require `lazy = true`.
"""
function ITOGraph(
        tt::ITOTermTable{I}, N::Int; lazy::Bool = true, jordan::Bool = false,
        channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    M = nterms(tt)
    (lazy || isempty(channels)) ||
        throw(ArgumentError("exponentially decaying channels need the lazy sweep"))
    sufid = _suffix_ids(tt)
    relsufid = _rel_suffix_ids(tt)
    firstsite = _first_sites(tt)
    lastent = maximum(c -> lastentry(c, N), channels; init = 0)

    rrepr = Int[]
    rcur = Int[]
    rbond = I[]
    rchan = Int[]
    rstate = Int[]
    inserted = falses(M)
    lefts = LeftVertex{I}[]
    radj = Vector{Int}[]
    wadj = Vector{ComplexF64}[]
    buckets = Dictionary{ITOKey{I}, Int}()

    function bucket!(key::ITOKey{I})
        b = get(buckets, key, 0)
        if iszero(b)
            push!(lefts, LeftVertex{I}(1, key))
            push!(radj, Int[])
            push!(wadj, ComplexF64[])
            b = length(lefts)
            insert!(buckets, key, b)
        end
        return b
    end

    for t in 1:M
        (lazy && firstsite[t] > 1) && continue
        b = bucket!(_op_at_ito(tt, t, 1))
        push!(rrepr, t)
        push!(rcur, 1)
        push!(rbond, unit(I))
        push!(rchan, 0)
        push!(rstate, 0)
        inserted[t] = true
        push!(radj[b], length(rrepr))
        push!(wadj[b], tt.coeffs[t])
    end

    # channels whose anchor is site 1 enter right here, on the same start channel a term does
    for (ci, c) in enumerate(channels)
        isentry(c, 1, N) || continue
        b = bucket!(c.entrykey)
        push!(rrepr, 0)
        push!(rcur, arity(tt) + 1)
        push!(rbond, c.states[c.start].bond)
        push!(rchan, ci)
        push!(rstate, c.start)
        push!(radj[b], length(rrepr))
        push!(wadj[b], c.coeff)
    end

    # pending bookkeeping: which site each uninserted term enters at, and its pre-start signature
    # (its full factor list, with the trivial running charge) so a colliding class can promote it
    pend_at = [Int[] for _ in 1:N]
    pendbysig = Dictionary{Tuple{Int, I}, Int}()
    nremaining = 0
    for t in 1:M
        inserted[t] && continue
        nremaining += 1
        push!(pend_at[firstsite[t]], t)
        insert!(pendbysig, (sufid[1, t], unit(I)), t)   # injective: `sufid[1, t]` fixes the term
    end

    # the identity/start channel's left vertex — where the sentinel and every injected term or channel
    # entry hangs off. A `K=0` term shares this exact `(link, key)`, so the bucket may already exist.
    startleft = 0
    if nremaining > 0 || 1 < lastent
        startleft = bucket!(ITOKey{I}(passthrough(I), unit(I), 1))
    end

    cap = length(rrepr) + 1   # right-vertex ids are at most one sentinel beyond the live ones
    return ITOGraph{I}(
        tt, N, arity(tt), sufid, relsufid, channels, lastent,
        rrepr, rcur, rbond, rchan, rstate, lefts, radj, wadj, 1,
        lazy, firstsite, pend_at, pendbysig, inserted, nremaining, 0, startleft, 0, 0, 0, 0, jordan,
        zeros(Int, cap), zeros(Int, cap), zeros(Int, cap), zeros(Int, cap),
        Dictionary{Tuple{Int, I}, Int}()
    )
end

# Phase 1: suffix-merge the right vertices, i.e. group those "equal from site i+1 on". Each live right
# vertex advances its cursor and hands over its two-word suffix signature; identical signatures merge.
# `Θ(live)` with `O(1)` per vertex, and — unlike the sorted-order/lcp scheme it replaces — indifferent
# to right vertices being created mid-sweep. Returns `remap[old_right_id] -> new_right_id` and updates
# `g.rrepr`/`g.rcur`/`g.rbond` in place.
function _suffix_merge!(g::ITOGraph{I}, i::Int) where {I}
    R = length(g.rrepr)
    groups = g.siggroups
    empty!(groups)
    remap = g.remap
    length(remap) < R && resize!(remap, R)

    newrepr = Int[]
    newcur = Int[]
    newbond = I[]
    newchan = Int[]
    newstate = Int[]
    for r in 1:R
        sig = _signature!(g, r, i)
        b = get(groups, sig, 0)
        if iszero(b)
            push!(newrepr, g.rrepr[r])
            push!(newcur, g.rcur[r])
            push!(newbond, g.rbond[r])
            push!(newchan, g.rchan[r])
            push!(newstate, g.rstate[r])
            b = length(newrepr)
            insert!(groups, sig, b)
        end
        remap[r] = b
    end

    g.rrepr = newrepr
    g.rcur = newcur
    g.rbond = newbond
    g.rchan = newchan
    g.rstate = newstate
    return remap
end

# Apply a right-vertex remap to every left vertex's adjacency, summing weights of edges that now land
# on the same merged right vertex. In place, via a scratch `rid -> position` table: `Θ(deg)` per left
# vertex with no allocation and no dictionary. The surviving order is first-encounter, which is
# deterministic; nothing downstream (matching, König, connected components) needs it sorted.
function _merge_edges!(g::ITOGraph, lv::Int, remap)
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
    # Drop edges whose accumulated weight cancelled to zero. They are not edges of the bipartite
    # graph and must not reach the cover, which would otherwise spend a bond index on them — the dense
    # predecessor fed `(!iszero).(coeff)` to `min_vertex_cover_bipartite` and so excluded them too.
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

# `slot` is all-zero on entry and on exit (every touched entry is reset above), so only newly grown
# space needs clearing — keeping the merge `O(deg)` rather than `O(nV)` per left vertex.
function _grow_scratch!(g::ITOGraph, nV::Int)
    for scratch in (g.slot, g.vlocal, g.firstleft)
        if length(scratch) < nV
            n0 = length(scratch)
            resize!(scratch, nV)
            fill!(view(scratch, (n0 + 1):nV), 0)
        end
    end
    return g
end

function _apply_remap!(g::ITOGraph, remap::Vector{Int}, nVnew::Int)
    _grow_scratch!(g, nVnew)
    for lv in eachindex(g.lefts)
        _merge_edges!(g, lv, remap)
    end
    return g
end

"""
    _promote_pending!(g, i)

Insert every not-yet-started term whose suffix class *coincides* with a live one at bond `i`.

This is the subtle half of lazy insertion. `_op_at_ito` fills idle sites with a pass-through carrying
the **running** bond charge, so a started term whose accumulated charge has fused back to `unit(I)` is
indistinguishable, over its idle sites, from a term that has not started yet. If its remaining factors
then coincide with the whole content of a pending term, the two suffix classes are genuinely equal and
the eager sweep merges them — covering the shared right vertex instead of spending a bond index. That
merge has to be reproduced, and at the *earliest* bond where it applies; suffix-equality-from-`i+1` is
monotone in `i`, so probing every bond finds it exactly once.

Cost: one hash probe per live right vertex (`pendbysig` is keyed injectively by a term's pre-start
signature, so a hit names a single term).
"""
function _promote_pending!(g::ITOGraph{I}, i::Int) where {I}
    (g.nremaining > 0 && !isempty(g.pendbysig)) || return g
    lv = g.startleft
    iszero(lv) && _invariant("pending terms with no start channel to inject them on")
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
    # the start channel may already have carried an edge to this class (a term promoted at an earlier
    # bond whose class has since merged), so parallel edges have to be summed
    promoted && _merge_edges!(g, lv, nothing)
    return g
end

"""
    _canonicalise_rights!(g, i)

Renumber the live right vertices into canonical order — ascending in their translation-invariant name
[`_rdesc`](@ref) — and sort every adjacency list to match.

This is what makes the sweep a deterministic function of the *canonical* graph rather than of its
construction history. Hopcroft–Karp's matching, and therefore König's cover, depend on the order the
adjacency lists are scanned in; without canonicalisation that order is first-encounter (`_merge_edges!`),
so two bonds posing isomorphic problems can answer them differently. On a finite chain that is
invisible (any minimum cover is as good as any other). On a periodic lattice it is fatal: the bond
basis then converges only up to a permutation of
itself, and the extracted unit cell does not close. Everything downstream is driven off right-vertex
ids and left-vertex order — `bipartite_connected_components` returns components in first-left-vertex
order with ascending ids — so canonical ids here plus canonical bond-index order in `_at_site!` pin
the whole sweep.

Called before the sentinel is attached, so every id in play is a real class.
"""
function _canonicalise_rights!(g::ITOGraph{I}, i::Int) where {I}
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

# Phases 1 & 2, shared by every graph-sweep strategy: suffix-merge the right vertices, apply the remap
# to the adjacency, promote any pending term whose class just became live, attach the sentinel that
# stands in for the still-pending terms, and record `g.firstleft` (first incident left vertex per right
# vertex) while checking that every right vertex is pure in the incoming bond charge — the
# block-diagonality invariant, here in `Θ(E)` off the sparse adjacency (one comparison per edge).
#
# The sentinel is *not* a term: it takes the right-vertex id one past the real ones, so everything that
# iterates `g.rrepr` skips it automatically and it is discarded (rather than forwarded) each bond.
# Returns `(nU, nV)`.
function _prepare_bond!(g::ITOGraph{I}, i::Int) where {I}
    g.rsent = 0
    remap = _suffix_merge!(g, i)
    nV = length(g.rrepr)
    _apply_remap!(g, remap, nV)
    _promote_pending!(g, i)
    _canonicalise_rights!(g, i)

    # the finish class: every factor placed (exhausted suffix) at the trivial running charge. The
    # suffix merge just made signatures unique, so there is at most one such right vertex.
    g.rfinish = 0
    @inbounds for r in eachindex(g.rrepr)
        # `_isexhausted` rather than a bare `sufid` lookup: a channel's right vertex has no
        # representative term (`rrepr == 0`), so indexing the table with it is out of bounds — and the
        # synthetic class a channel exits onto is exhausted without having a cursor at all
        if _isexhausted(g, r) && g.rbond[r] == unit(I)
            g.rfinish = r
            break
        end
    end

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
                    _invariant("bond index not sector-pure (block-diagonality violated)")
            end
        end
    end

    return nU, nV
end

# Phase 5, shared by both backends: build the next graph reusing the SAME (persistent) right vertices
# and tagging fresh left vertices with the outgoing bond index `j` as their `link`, bucketed by the
# next-site key `op@(i+1)` of the right vertex. `nextedges_global[j]` is the list of `(right_vertex,
# weight)` edges the outgoing bond index `j` forwards. Mutates `g` into the graph for bond `i → i+1`.
#
# The next-site key depends only on the right vertex, so it is computed once per right vertex rather
# than once per edge (the same right vertex is typically forwarded by many outgoing bond indices).
#
# This is also where lazy insertion *injects*: a term whose first active site is `i+1` gets its right
# vertex here, hanging off the start channel with the term's coefficient as the edge weight — exactly
# the weight the eager sweep would have been carrying along that channel since bond 0. An exponentially
# decaying channel enters the same way, but at *every* site matching its phase rather than once.
#
# A geometric right vertex is the one place where a class has more than one successor: its automaton
# state carries a transition per string letter (continue, weight `λ·c`) and, at `δ = 1`, one for the exit
# factor. Each becomes its own left-vertex bucket, so a single forwarded edge fans out into several — the
# continuation is what puts `λ · pass-through` on the channel's diagonal, and the exit is what lets it
# reach the done channel. Transitions whose target could no longer complete inside `1:N` are dropped:
# those translates do not exist on this lattice, and pruning them is what keeps the last bond of a finite
# chain one-dimensional.
function _build_next_graph!(
        g::ITOGraph{I}, i::Int, nout::Int,
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}}
    ) where {I}
    # `_next_key` is right for a term class *and* for the synthetic exhausted class (whose cursor is
    # past the end, so it returns the bare pass-through at the running charge — the class's own
    # transition onto itself). Only a live channel state has several successors, handled below.
    nextkeys = [_next_key(g, r, i) for r in eachindex(g.rrepr)]
    buckets = Dictionary{Tuple{Int, ITOKey{I}}, Int}()
    next_lefts = LeftVertex{I}[]
    next_radj = Vector{Int}[]
    next_wadj = Vector{ComplexF64}[]

    function bucket!(j::Int, key::ITOKey{I})
        b = get(buckets, (j, key), 0)
        if iszero(b)
            push!(next_lefts, LeftVertex{I}(j, key))
            push!(next_radj, Int[])
            push!(next_wadj, ComplexF64[])
            b = length(next_lefts)
            insert!(buckets, (j, key), b)
        end
        return b
    end

    # a class the sweep has to reach may not exist as a right vertex yet: the automaton's successor
    # states and the exhausted class a channel exits onto are created here, on demand, and reused
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
        iszero(state) && return get!(() -> _push_geometric!(g, -1, 0, bond), exhausted, bond)
        return get!(() -> _push_geometric!(g, chan, state, bond), geom, (chan, state))
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
        # Guarded here rather than inside the loop below: `g.startleft` is rebuilt off `startidx` even
        # when no term enters at `i+1`, so a missing start channel has to be caught either way.
        # `nremaining > 0` (or an entry still to come) means the sentinel existed at this bond, which
        # forces a start channel. Otherwise every term is already inserted and no channel can still
        # enter, so `pend_at[i+1]` and the channel loop below are both no-ops.
        ((iszero(g.nremaining) && i + 1 > g.lastentry) || !iszero(startidx)) ||
            _invariant("terms remain to the right of site $i with no start channel to enter on")
        for t in g.pend_at[i + 1]
            g.inserted[t] && continue         # already promoted into a colliding class
            g.inserted[t] = true
            g.nremaining -= 1
            push!(g.rrepr, t)
            push!(g.rcur, 1)
            push!(g.rbond, unit(I))
            push!(g.rchan, 0)
            push!(g.rstate, 0)
            b = bucket!(startidx, g.tt.keys[1, t])
            push!(next_radj[b], length(g.rrepr))
            push!(next_wadj[b], g.tt.coeffs[t])
        end
        # every channel whose phase matches site `i+1` enters there, on the same start channel — and
        # keeps doing so at every later matching site, which is why it is live at every bulk bond
        for (ci, c) in enumerate(g.channels)
            isentry(c, i + 1, g.N) || continue
            b = bucket!(startidx, c.entrykey)
            st = c.states[c.start]
            push!(next_radj[b], geomvertex!(ci, c.start, st.bond))
            push!(next_wadj[b], c.coeff)
        end
        # the start channel has to survive even when it forwards nothing, so that the next bond's
        # sentinel (and the terms or channel entries after that) still have a left vertex to hang off
        g.startleft = (g.nremaining > 0 || i + 1 < g.lastentry) ?
            bucket!(startidx, ITOKey{I}(passthrough(I), unit(I), 1)) : 0
    end

    # the finish channel forwards exactly one edge — to the exhausted class, whose next-site key is
    # the trivial pass-through — so its continuation is a single left vertex, and that left vertex has
    # no other neighbour. Looking it up here is what lets the next bond recognise its own finish index.
    g.finishleft = 0
    if !iszero(g.finishidx)
        g.finishleft = get(buckets, (g.finishidx, ITOKey{I}(passthrough(I), unit(I), 1)), 0)
        iszero(g.finishleft) &&
            _invariant("the finish channel at bond $i forwards nothing to site $(i + 1)")
    end

    g.lefts = next_lefts
    g.radj = next_radj
    g.wadj = next_wadj
    g.nlinks = nout
    return g
end

# One site step (ITensor's `at_site!`), five phases: (1) suffix-merge the right vertices; (2)
# connected components / bond assembly; (3) the strategy's bond-basis choice; (4) assemble the bond;
# (5) build the next graph, reusing the same right vertices and tagging fresh left vertices with the
# outgoing bond index as `link`. Phases 1, 2 and 5 are strategy-independent; `_bond_basis!` is the
# plug point and covers 3 & 4. Returns `(Ws_i, secW_i)` and mutates `g` into the graph for bond
# `i → i+1`.
function _at_site!(g::ITOGraph{I}, i::Int, strategy::GraphStrategy = VertexCover()) where {I}
    nU, nV = _prepare_bond!(g, i)
    g.startidx = 0
    g.finishidx = 0
    nout, site_dict, secW, nextedges_global = _bond_basis!(g, i, nU, nV, strategy)
    Ws_i = sparse_from_dict(site_dict, (g.nlinks, nout))
    i < g.N && _build_next_graph!(g, i, nout, nextedges_global)
    return Ws_i, secW, g.startidx, g.finishidx
end

# Lazy right-vertex insertion pays off only when the sweep is driven off the sparse adjacency: the
# SVD strategy materialises a dense per-bond coefficient matrix anyway, so laziness would buy nothing
# while adding a sentinel column for the SVD to carry.
_graph_lazy(::VertexCover) = true
_graph_lazy(::SequentialSVD) = false

# Resolve `trunc === nothing` (the lossless default) once per sweep rather than once per bond.
_resolve_trunc(s::BondStrategy) = s

"""
    _irrep_graph_sweep(tt::ITOTermTable{I}, N, strategy) -> (Ws, bondsectors)

The persistent-graph reduced-MPO sweep, run with a bond-basis `strategy` ([`VertexCover`](@ref) or
[`SequentialSVD`](@ref)). Produces the `(Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}},
bondsectors::Vector{Vector{I}})` contract that `mpo_terms` / `irrep_mpo_tensors` consume.

`channels` are lowered exponentially decaying interactions ([`_lower_channels`](@ref)), which live
alongside the term table rather than in it; a model may consist of nothing else.
"""
function _irrep_graph_sweep(
        tt::ITOTermTable{I}, N::Int, strategy::GraphStrategy;
        channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    Ws, bondsectors, _, _ = _irrep_graph_channels(tt, N, strategy, false; channels)
    return (Ws, bondsectors)
end

"""
    _irrep_graph_channels(tt::ITOTermTable{I}, N, strategy, jordan) -> (Ws, bondsectors, starts, finishes)

[`_irrep_graph_sweep`](@ref) plus the per-bond identity-channel indices that Jordan emission
(jordanmpo.jl) needs: `starts[i]` / `finishes[i]` are the bond indices of the start ("nothing placed
yet") and finish ("everything placed") channels at the bond to the right of site `i`, or `0` where the
cover did not spend an index on that channel. Only [`VertexCover`](@ref) has them — an SVD bond basis
is a *mixture* of prefix states, in which neither channel is a basis vector — so
[`SequentialSVD`](@ref) reports `0` throughout and every channel gets padded.

`jordan = true` also forces the finish class into every cover ([`_force_finish!`](@ref)), which is why
this is a separate entry point rather than extra return values on `_irrep_graph_sweep`: it can change
the bond basis, and `irrep_mpo` promises the unconstrained minimum.
"""
function _irrep_graph_channels(
        tt::ITOTermTable{I}, N::Int, strategy::GraphStrategy, jordan::Bool;
        channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    LOp = SiteOperator{I}
    (nterms(tt) == 0 && isempty(channels)) &&
        return (SparseMatrixCSC{LOp, Int}[], Vector{I}[], Int[], Int[])

    strategy = _resolve_trunc(strategy)
    g = ITOGraph(tt, N; lazy = _graph_lazy(strategy), jordan, channels)
    Ws = Vector{SparseMatrixCSC{LOp, Int}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    starts = zeros(Int, N)
    finishes = zeros(Int, N)
    for i in 1:N
        Ws[i], bondsectors[i], starts[i], finishes[i] = _at_site!(g, i, strategy)
    end
    return (Ws, bondsectors, starts, finishes)
end

"""
    _irrep_sweep(tt::ITOTermTable{I}, N, strategy::BondStrategy) -> (Ws, bondsectors)

Run the reduced-MPO compression of `tt` over `N` sites with the given bond-basis strategy. This is
the single entry point `irrep_mpo` (irrepmpo.jl) dispatches to; the strategy decides whether that is
the persistent-graph sweep or the independent per-bond pass.
"""
_irrep_sweep(
    tt::ITOTermTable{I}, N::Int, strategy::GraphStrategy;
    channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
) where {I} = _irrep_graph_sweep(tt, N, strategy; channels)

"""
    _irrep_channels(tt::ITOTermTable, N, strategy::BondStrategy) -> (Ws, bondsectors, starts, finishes)

[`_irrep_sweep`](@ref) with the identity-channel indices [`_irrep_graph_channels`](@ref) documents,
and with the finish class forced into the cover. The independent-SVD pass has no persistent bond
identity at all, so it reports none and Jordan emission pads every channel.
"""
_irrep_channels(tt::ITOTermTable, N::Int, s::GraphStrategy) = _irrep_graph_channels(tt, N, s, true)
