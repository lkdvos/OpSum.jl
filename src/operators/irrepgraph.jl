# Persistent bipartite-graph MPO construction for the ITO automaton
# =================================================================
# A port of ITensorMPOConstruction.jl's persistent-graph + `at_site!` sweep architecture onto
# OpSum's non-abelian (TensorKit `Sector`) ITO machinery. It builds the *same* reduced MPO as the
# transient-frontier sweep `_irrep_bipartite` (irreptermtable.jl) — same per-bond-sector dimensions,
# same `(Ws, bondsectors)` contract — but keeps an explicit bipartite graph handed from one site step
# to the next, instead of re-materialising every strand's suffix path each bond.
#
# Cost is `Θ(M·K)` to intern the suffix classes plus `Θ(Σ_terms span)` for the sweep, i.e. linear in N
# for a finite-range model. Two things buy that, and both are load-bearing: suffix classes are named by
# an interned `O(1)` signature rather than a materialised path (`_suffix_ids`, `_signature!`), and a
# term's right vertex is created only once it is reachable (`_promote_pending!` and the injection in
# `_build_next_graph!`). See research/persistent-graph-mpo.md §2 — in particular §2.2, whose
# pending-versus-started class collision is the invariant a change here is most likely to break.
#
# Non-abelian mapping (see research/itensor-mpograph-construction.md §9):
# * Right vertices  = suffix classes of the `ITOTermTable`, entering at their term's first active site
#   and thereafter only merging. ITensor's "term + additive QN flux" becomes "term + running fusion
#   charge", carried in each site's `ITOKey.bond` (a fusion *outcome*, not a sum).
# * Left vertices   = `LeftVertex(link, key::ITOKey)` — the incoming bond index plus the on-site ITO
#   key applied here. Rebuilt each site.
# * The graph sweep stays SCALAR: reduced coefficients are `ComplexF64`, min-vertex-cover runs on the
#   scalar adjacency. Non-abelian structure enters only (a) in what makes a bond state distinct (the
#   augmented `ITOKey` → `bondsectors`) and (b) later, at tensor assembly (`irrep_mpo_tensors`).
# * Connected components are pure in the outgoing bond charge (`ITOKey.bond`) — asserted, exactly as
#   the transient sweep does.
#
# Supported arity K ∈ {0,1,2}; the `ITOKey.vertex` multiplicity label is threaded through
# `LeftVertex.key` but is `1` throughout the multiplicity-free scope, so the reduced `(Ws,
# bondsectors)` contract is unchanged. GenericFusion multi-channel and fermionic (graded) sectors are
# out of scope (the `LeftVertex` fermion/JW slot from ITensor is intentionally omitted).

using TensorKit: Sector
using SparseArrays: SparseMatrixCSC

"""
    LeftVertex{I}

A left (prefix) vertex of the persistent ITO graph: it entered the current site on incoming bond
index `link` and applies the on-site ITO key `key = (op, bond, vertex)` here. The non-abelian
analogue of ITensorMPOConstruction's `LeftVertex(link, op_id, needs_JW)`; the fermion/JW-string slot
is intentionally omitted (the irrep track has no fermions — a documented follow-up).
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
  *contiguous column suffix* of `tt` — see [`_suffix_ids`](@ref);
* persistent right-vertex state (shrinks via the suffix-merge): `rrepr` (representative term id) plus
  the monotone cursor `rcur`/`rbond` that turns "the suffix path from site `i+1`" into an `O(1)`
  two-word signature;
* the current bipartite graph: `lefts`, `radj`, `wadj`, and `nlinks` (incoming bond dimension);
* lazy-insertion state (see [`_promote_pending!`](@ref)): `lazy`, `firstsite`, `pend_at`, `pendbysig`,
  `inserted`, `nremaining`, and the per-bond `rsent` (sentinel right-vertex id, 0 if none),
  `startleft` (the left vertex the sentinel hangs off, 0 if none) and `startidx` (the outgoing bond
  index of the start channel);
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
    rrepr::Vector{Int}     # right vertex -> representative term id
    rcur::Vector{Int}      # right vertex -> first column j with sites[j, rrepr] > current site
    rbond::Vector{I}       # right vertex -> running bond charge just past the current site
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
    slot::Vector{Int}      # scratch: right-vertex id -> position within one left vertex's adjacency
    vlocal::Vector{Int}    # scratch: right-vertex id -> component-local index
    firstleft::Vector{Int} # right-vertex id -> first incident left vertex (0 if isolated)
    remap::Vector{Int}     # scratch: old right-vertex id -> merged right-vertex id
    siggroups::Dictionary{Tuple{Int, I}, Int}  # scratch: suffix signature -> merged right-vertex id
end

"""
    _suffix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Intern every term's *contiguous column suffixes*. `tt.sites` columns are ascending and zero-padded, so
a term's active factors at sites `> i` are always a suffix `j₀:K` of its column; `sufid[j, t]` is a
dense integer id for the factor list `j:K` of term `t`, with `0` for the exhausted suffix. Built
bottom-up in `Θ(M·K)` — the replacement for materialising a length-`N` path per term.

Equality of ids is equality of the remaining factor list. That is *not* by itself equality of the
suffix path: `_op_at_ito` fills idle sites with a pass-through carrying the running bond charge, and
the idle sites *before* the first remaining factor carry the charge accumulated so far. So a suffix
path is identified by the pair `(sufid[j₀, t], running bond charge)` — see `_signature`.
"""
function _suffix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    sufid = zeros(Int, K + 1, M)      # row K+1 and every padded position stay 0 == exhausted
    intern = Dictionary{Tuple{Int, ITOKey{I}, Int}, Int}()
    nid = 0
    for t in 1:M
        for j in K:-1:1
            s = tt.sites[j, t]
            iszero(s) && continue    # padding: the suffix from here on is exhausted
            trans = (s, tt.keys[j, t], sufid[j + 1, t])
            id = get(intern, trans, 0)
            if iszero(id)
                nid += 1
                id = nid
                insert!(intern, trans, id)
            end
            sufid[j, t] = id
        end
    end
    return sufid
end

# Advance right vertex `r`'s cursor past every factor at a site `<= i`, accumulating the running bond
# charge, and return its suffix signature `(interned remaining-factor list, running bond charge)`.
# Amortised `O(1)`: each cursor advances at most `K` times over the whole sweep.
function _signature!(g::ITOGraph{I}, r::Int, i::Int) where {I}
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
    ITOGraph(tt::ITOTermTable{I}, N; lazy = true) -> ITOGraph{I}

Seed the persistent graph (mirrors ITensor's `MPOGraph(os)`): intern the column suffixes, create the
right vertices, and bucket the initial left vertices by the site-1 key against the single
left-boundary link. `Θ(M·K)`, with no length-`N` path and no sort.

With `lazy = true` (the default) only terms *active at site 1* get a right vertex; the rest are
represented collectively by a sentinel right vertex on the identity/start channel and are inserted
when they become reachable (`_promote_pending!` / the injection in `_build_next_graph!`). That is what
makes the sweep cost `Θ(Σ_terms span)` instead of `Θ(N·M)`. `lazy = false` seeds every term eagerly,
which is the form `_irrep_graph_svd` uses (its per-bond dense SVD dominates anyway).
"""
function ITOGraph(tt::ITOTermTable{I}, N::Int; lazy::Bool = true) where {I}
    M = nterms(tt)
    sufid = _suffix_ids(tt)
    firstsite = _first_sites(tt)

    rrepr = Int[]
    rcur = Int[]
    rbond = I[]
    inserted = falses(M)
    lefts = LeftVertex{I}[]
    radj = Vector{Int}[]
    wadj = Vector{ComplexF64}[]
    buckets = Dictionary{ITOKey{I}, Int}()

    for t in 1:M
        (lazy && firstsite[t] > 1) && continue
        key1 = _op_at_ito(tt, t, 1)
        b = get(buckets, key1, 0)
        if iszero(b)
            push!(lefts, LeftVertex{I}(1, key1))
            push!(radj, Int[])
            push!(wadj, ComplexF64[])
            b = length(lefts)
            insert!(buckets, key1, b)
        end
        push!(rrepr, t)
        push!(rcur, 1)
        push!(rbond, unit(I))
        inserted[t] = true
        push!(radj[b], length(rrepr))
        push!(wadj[b], tt.coeffs[t])
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

    # the identity/start channel's left vertex — where the sentinel and every injected term hang off.
    # A `K=0` term shares this exact `(link, key)`, so the bucket may already exist.
    startleft = 0
    if nremaining > 0
        key0 = ITOKey{I}(passthrough(I), unit(I), 1)
        startleft = get(buckets, key0, 0)
        if iszero(startleft)
            push!(lefts, LeftVertex{I}(1, key0))
            push!(radj, Int[])
            push!(wadj, ComplexF64[])
            startleft = length(lefts)
        end
    end

    cap = M + 1   # right-vertex ids are at most one sentinel beyond the real ones
    return ITOGraph{I}(
        tt, N, arity(tt), sufid, rrepr, rcur, rbond, lefts, radj, wadj, 1,
        lazy, firstsite, pend_at, pendbysig, inserted, nremaining, 0, startleft, 0,
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
    for r in 1:R
        sig = _signature!(g, r, i)
        b = get(groups, sig, 0)
        if iszero(b)
            push!(newrepr, g.rrepr[r])
            push!(newcur, g.rcur[r])
            push!(newbond, g.rbond[r])
            b = length(newrepr)
            insert!(groups, sig, b)
        end
        remap[r] = b
    end

    g.rrepr = newrepr
    g.rcur = newcur
    g.rbond = newbond
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

# Per-component minimum-vertex-cover backend (the VC path). Given a connected component `(us, vs)`
# (global left/right vertex ids), it chooses the component's bond basis via
# `min_vertex_cover_bipartite` and returns, for that component: `rank`, `blocks`
# (`(incoming_link, local_bond_index, localop)`), `nextedges` (per local bond index, the
# `(right_vertex_id, weight)` edges to forward; empty at the last site), `secs` (the bond charge per
# local index) and `startidx` (the local index of the identity/start channel, 0 if this component does
# not hold it). Reproduces the covered-U / covered-V coefficient-flow of `_irrep_bipartite`.
#
# Everything is driven off the sparse adjacency `g.radj`/`g.wadj` plus `g.firstleft`, so the cost is
# `Θ(E_component)` — no `|us| × |vs|` matrix is ever formed, and neither the covered-left forwarding
# nor the covered-right folding scans the opposite side.
#
# The sentinel needs no special-casing in the cover, only in the *forwarding*. It is a degree-1 right
# vertex, and König's construction never covers one: in a maximum matching `L₀` is matched (or `L₀`–
# sentinel would augment), and the sentinel can only be reached from `L₀` — via their matching edge,
# which the forward search skips, or as a free vertex, which would complete an augmenting path. So
# `L₀` is always covered-left and emits the bare pass-through letter into `(L₀.link, m₀)`.
#
# The covered-right sentinel case below is therefore unreachable. It is kept because it costs two
# comparisons and is the exact dual: an uncovered `L₀` folds `passthrough × 1` into that same block, so
# a future change to the cover construction cannot silently produce a bond with no identity channel.
function _vc_component(
        g::ITOGraph{I}, us::Vector{Int}, vs::Vector{Int}, i::Int
    ) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    N = g.N
    nus, nvs = length(us), length(vs)

    # component-local right-vertex numbering (every neighbour of a `us` vertex lies in `vs`)
    vlocal = g.vlocal
    @inbounds for p in 1:nvs
        vlocal[vs[p]] = p
    end
    localadj = Vector{Vector{Int}}(undef, nus)
    @inbounds for k in 1:nus
        radj = g.radj[us[k]]
        localadj[k] = Int[vlocal[rid] for rid in radj]
    end

    cUbits, cVbits = min_vertex_cover_bipartite(localadj, nus, nvs)
    cU = findall(cUbits)
    cV = findall(cVbits)
    nleft = length(cU)
    rank = nleft + length(cV)

    blocks = Tuple{Int, Int, LOp}[]
    nextedges = [Tuple{Int, ComplexF64}[] for _ in 1:rank]
    secs = Vector{I}(undef, rank)
    startidx = 0

    # covered-left vertices → local bond indices 1 … nleft ("a term starts its operator here")
    for (m, lu) in enumerate(cU)
        iu = us[lu]
        lv = g.lefts[iu]
        secs[m] = lv.key.bond
        iu == g.startleft && (startidx = m)
        if i == N
            @assert iszero(g.rsent) "the sentinel must be gone by the last site"
            push!(blocks, (lv.link, m, lv.key.op * sum(g.wadj[iu]; init = zero(ComplexF64))))
        else
            push!(blocks, (lv.link, m, convert(LOp, lv.key.op)))
            edges = nextedges[m]
            for (rid, w) in zip(g.radj[iu], g.wadj[iu])
                (rid == g.rsent || iszero(w)) && continue   # the sentinel is regenerated, not carried
                push!(edges, (rid, w))
            end
        end
    end

    # covered-right vertices → local bond indices nleft+1 … rank ("a shared suffix flows through")
    # `bondof[p]` is the bond index a covered right vertex takes, 0 if it is not covered.
    bondof = zeros(Int, nvs)
    for (p, lvv) in enumerate(cV)
        m = nleft + p
        iv = vs[lvv]
        bondof[lvv] = m
        # the component is pure in the bond charge (asserted in `_prepare_bond!`), so any incident
        # left vertex gives it
        secs[m] = g.lefts[g.firstleft[iv]].key.bond
        if iv == g.rsent
            @assert iszero(startidx) "start channel covered on both sides"
            startidx = m                        # the sentinel *is* the start channel here
        elseif i != N
            push!(nextedges[m], (iv, one(ComplexF64)))
        end
    end

    # every neighbour of an *uncovered* left vertex is a covered right vertex (otherwise that edge
    # would be uncovered), so one pass over the uncovered lefts' edges folds all the coefficients
    coveredU = falses(nus)
    coveredU[cU] .= true
    for k in 1:nus
        coveredU[k] && continue
        iu = us[k]
        lv = g.lefts[iu]
        for (rid, w) in zip(g.radj[iu], g.wadj[iu])
            iszero(w) && continue
            m = bondof[vlocal[rid]]
            @assert !iszero(m) "edge left uncovered by the minimum vertex cover"
            push!(blocks, (lv.link, m, lv.key.op * w))
        end
    end

    return rank, blocks, nextedges, secs, startidx
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
    @assert !iszero(lv) "pending terms with no start channel to inject them on"
    promoted = false
    for r in eachindex(g.rrepr)
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

# Phases 1 & 2, shared by both backends: suffix-merge the right vertices, apply the remap to the
# adjacency, promote any pending term whose class just became live, attach the sentinel that stands in
# for the still-pending terms, and record `g.firstleft` (first incident left vertex per right vertex)
# while asserting that every right vertex is pure in the incoming bond charge — the block-diagonality
# invariant `_irrep_bipartite` also checks, here in `Θ(E)` off the sparse adjacency.
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

    if g.nremaining > 0
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
                @assert g.lefts[firstleft[rid]].key.bond == bond "bond index not sector-pure (block-diagonality violated)"
            end
        end
    end

    return nU, nV
end

# The dense `coeff[u, v]` matrix, materialised from the sparse adjacency. Only the SVD backend needs
# it (its cost is dominated by the dense per-bond SVD anyway); the VC backend stays sparse.
function _dense_bond_matrix(g::ITOGraph, nU::Int, nV::Int)
    coeff = zeros(ComplexF64, nU, nV)
    for iu in 1:nU
        for (rid, w) in zip(g.radj[iu], g.wadj[iu])
            coeff[iu, rid] += w
        end
    end
    return coeff
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
# the weight the eager sweep would have been carrying along that channel since bond 0.
function _build_next_graph!(
        g::ITOGraph{I}, i::Int, nout::Int,
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}}
    ) where {I}
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

    for j in 1:nout
        for (rid, w) in nextedges_global[j]
            b = bucket!(j, nextkeys[rid])
            push!(next_radj[b], rid)
            push!(next_wadj[b], w)
        end
    end

    if g.lazy
        startidx = g.startidx
        # Guarded here rather than inside the loop below: `g.startleft` is rebuilt off `startidx` even
        # when no term enters at `i+1`, so a missing start channel has to be caught either way.
        # `nremaining > 0` means the sentinel existed at this bond, which forces a start channel. When
        # it is 0 every term is already inserted, `pend_at[i+1]` is a no-op and no channel is needed.
        @assert iszero(g.nremaining) || !iszero(startidx) "terms remain to the right of site $i with no start channel to enter on"
        for t in g.pend_at[i + 1]
            g.inserted[t] && continue         # already promoted into a colliding class
            g.inserted[t] = true
            g.nremaining -= 1
            push!(g.rrepr, t)
            push!(g.rcur, 1)
            push!(g.rbond, unit(I))
            b = bucket!(startidx, g.tt.keys[1, t])
            push!(next_radj[b], length(g.rrepr))
            push!(next_wadj[b], g.tt.coeffs[t])
        end
        # the start channel has to survive even when it forwards nothing, so that the next bond's
        # sentinel (and the terms after that) still have a left vertex to hang off
        g.startleft = g.nremaining > 0 ?
            bucket!(startidx, ITOKey{I}(passthrough(I), unit(I), 1)) : 0
    end

    g.lefts = next_lefts
    g.radj = next_radj
    g.wadj = next_wadj
    g.nlinks = nout
    return g
end

# One site step (ITensor's `at_site!`), five phases: (1) suffix-merge the right vertices; (2)
# connected components; (3) per-component vertex-cover backend; (4) assemble the bond (concatenate
# component ranks, collecting charges + forwarded edges); (5) build the next graph, reusing the same
# right vertices and tagging fresh left vertices with the outgoing bond index as `link`. Returns
# `(Ws_i, secW_i)` and mutates `g` into the graph for bond `i → i+1`.
function _at_site!(g::ITOGraph{I}, i::Int) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}

    nU, nV = _prepare_bond!(g, i)
    us_of_comp, vs_of_comp = bipartite_connected_components(g.radj, nV)

    # phases 3 & 4 — per-component cover, concatenate into the bond
    secW = I[]
    nextedges_global = Vector{Tuple{Int, ComplexF64}}[]
    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    offset = 0
    g.startidx = 0
    for (us, vs) in zip(us_of_comp, vs_of_comp)
        rank, blocks, nextedges, secs, startidx = _vc_component(g, us, vs, i)
        for (link, m, op) in blocks
            increaseindex!(site_dict, CartesianIndex(link, offset + m), op)
        end
        for m in 1:rank
            push!(secW, secs[m])
            push!(nextedges_global, nextedges[m])
        end
        if !iszero(startidx)
            @assert iszero(g.startidx) "the start channel appeared in more than one component"
            g.startidx = offset + startidx
        end
        offset += rank
    end
    nout = offset

    Ws_i = sparse_from_dict(site_dict, (g.nlinks, nout))
    i < g.N && _build_next_graph!(g, i, nout, nextedges_global)
    return Ws_i, secW
end

"""
    _irrep_graph_bipartite(tt::ITOTermTable{I}, N) -> (Ws, bondsectors)

Persistent-graph, minimum-vertex-cover reduced-MPO sweep — the ITensor `at_site!` port. Produces the
same `(Ws::Vector{SparseMatrixCSC{LocalOp{ComplexF64, IrrepOperator{I}}, Int}}, bondsectors::Vector{
Vector{I}})` contract as `_irrep_bipartite`, so `mpo_terms` / `irrep_mpo_tensors` consume it
unchanged.
"""
function _irrep_graph_bipartite(tt::ITOTermTable{I}, N::Int) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    nterms(tt) == 0 && return (SparseMatrixCSC{LOp, Int}[], Vector{I}[])

    g = ITOGraph(tt, N)
    Ws = Vector{SparseMatrixCSC{LOp, Int}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    for i in 1:N
        Ws[i], bondsectors[i] = _at_site!(g, i)
    end
    return (Ws, bondsectors)
end

# SVD site step (ITensor's `at_site!` with the QR/SVD backend, doc §6 "The QR backend"). Phases 1, 2
# and 5 are shared with the VC step; only the bond-basis choice differs: instead of a per-component
# minimum vertex cover, the WHOLE bond's scalar coefficient matrix is assembled as a charge-graded
# `TensorMap C : Ppre ← Psuf` (block-diagonal in the bond charge, so `svd_trunc` does the per-sector
# SVD *and* the global-across-sectors truncation at once, respecting quantum dimensions — matching
# `_irrep_svd`). Keeping `U` (left singular vectors) as the compressed bond basis: each outgoing bond
# index `m` is a linear combination `U[u, m]` of prefix states, emitting `key.op * U[u, m]` into the
# `(link, m)` block; the residual `R = S·Vᴴ` forwards the coefficient onto the next bond's edges
# (folded into the block at the last site). Mutates `g` into the graph for bond `i → i+1`.
function _svd_at_site!(g::ITOGraph{I}, i::Int, truncstrat) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    N = g.N

    nU, nV = _prepare_bond!(g, i)
    coeff = _dense_bond_matrix(g, nU, nV)

    # bond charge of each prefix (left) state and each suffix (right) state; sector-pure per column.
    # A right vertex may be orphaned (no incident edge) after a *truncation* dropped the singular
    # vector that coupled to it — such suffix classes are simply unreachable and carry no weight.
    uCharge = I[g.lefts[u].key.bond for u in 1:nU]
    vCharge = Vector{I}(undef, nV)
    vactive = falses(nV)
    for v in 1:nV
        conn = findall(!iszero, @view coeff[:, v])
        isempty(conn) && continue
        q = uCharge[first(conn)]
        @assert all(uCharge[u] == q for u in conn) "bond index not sector-pure (block-diagonality violated)"
        vCharge[v] = q
        vactive[v] = true
    end

    # per-state degeneracy index within its charge sector + per-sector multiplicities
    umult = Dict{I, Int}()
    udeg = zeros(Int, nU)
    for u in 1:nU
        udeg[u] = umult[uCharge[u]] = get(umult, uCharge[u], 0) + 1
    end
    vmult = Dict{I, Int}()
    vdeg = zeros(Int, nV)
    for v in 1:nV
        vactive[v] || continue
        vdeg[v] = vmult[vCharge[v]] = get(vmult, vCharge[v], 0) + 1
    end

    Ppre = Vect[I](umult)
    Psuf = Vect[I](vmult)
    C = zeros(ComplexF64, Ppre ← Psuf)
    for v in 1:nV, u in 1:nU
        iszero(coeff[u, v]) && continue
        block(C, uCharge[u])[udeg[u], vdeg[v]] += coeff[u, v]
    end

    U, S, Vt = svd_trunc(C; trunc = truncstrat)
    Wb = space(S, 1)                 # retained bond space (⊕ charge sectors, truncated multiplicities)
    R = S * Vt                       # Wb ← Psuf, forwards the coefficient onto the next bond

    r = sum(q -> dim(Wb, q), sectors(Wb); init = 0)
    secW = Vector{I}(undef, r)
    Umat = zeros(ComplexF64, nU, r)  # column m = compressed prefix basis vector
    Rmat = zeros(ComplexF64, r, nV)  # row m    = suffix expressed in the new basis
    col = 0
    for q in sectors(Wb)
        Ub = block(U, q)
        Rb = block(R, q)
        for m in 1:size(Ub, 2)
            col += 1
            secW[col] = q
            for u in 1:nU
                uCharge[u] == q && (Umat[u, col] = Ub[udeg[u], m])
            end
            for v in 1:nV
                (vactive[v] && vCharge[v] == q) && (Rmat[col, v] = Rb[m, vdeg[v]])
            end
        end
    end

    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    nextedges_global = [Tuple{Int, ComplexF64}[] for _ in 1:r]
    for m in 1:r
        for u in 1:nU
            w = Umat[u, m]
            iszero(w) && continue
            lv = g.lefts[u]
            if i == N
                # last site: the single trivial suffix folds R into the block
                increaseindex!(site_dict, CartesianIndex(lv.link, m), lv.key.op * (w * Rmat[m, 1]))
            else
                increaseindex!(site_dict, CartesianIndex(lv.link, m), lv.key.op * w)
            end
        end
        if i < N
            for v in 1:nV
                rw = Rmat[m, v]
                iszero(rw) || push!(nextedges_global[m], (v, rw))
            end
        end
    end

    Ws_i = sparse_from_dict(site_dict, (g.nlinks, r))
    i < N && _build_next_graph!(g, i, r, nextedges_global)
    return Ws_i, secW
end

"""
    _irrep_graph_svd(tt::ITOTermTable{I}, N, trunc) -> (Ws, bondsectors)

Persistent-graph SVD reduced-MPO sweep — the ITensor QR-backend port. Same `(Ws, bondsectors)`
contract as `_irrep_graph_bipartite` / `_irrep_svd`; each bond's compressed basis is
the left singular vectors of the charge-graded bond coefficient matrix, truncated globally across
sectors by `trunc` (a `MatrixAlgebraKit.TruncationStrategy`, or `nothing` for the lossless default).

**Lossless (`trunc === nothing`) this is at parity with `_irrep_svd`** — same per-sector bond
dimensions and same represented operator. Under **truncation the two diverge by design**: this is a
*sequential* left-to-right sweep (each bond is compressed in the basis left over from the bond before
it, à la ITensor's QR sweep), whereas `_irrep_svd` compresses every bond *independently* on the raw
prefix/suffix classes. Both are exact losslessly; under aggressive truncation the sequential sweep can
starve downstream bonds. The default `SVDBondAlgorithm` selector therefore routes to `_irrep_svd` to
preserve the pinned per-bond-independent truncation semantics; wiring this sequential variant behind a
selector (and choosing between the two truncation semantics) is a documented follow-up.
"""
function _irrep_graph_svd(tt::ITOTermTable{I}, N::Int, trunc) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    nterms(tt) == 0 && return (SparseMatrixCSC{LOp, Int}[], Vector{I}[])

    truncstrat = something(trunc, trunctol(rtol = eps(Float64)))
    # eager seeding: this backend materialises a dense per-bond coefficient matrix anyway, so lazy
    # insertion would buy nothing while adding a sentinel column the SVD would have to carry
    g = ITOGraph(tt, N; lazy = false)
    Ws = Vector{SparseMatrixCSC{LOp, Int}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    for i in 1:N
        Ws[i], bondsectors[i] = _svd_at_site!(g, i, truncstrat)
    end
    return (Ws, bondsectors)
end
