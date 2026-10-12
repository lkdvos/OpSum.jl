# Reduced-MPO sweeps for the ITO automaton. `VertexCover`/`SequentialSVD` run the persistent
# bipartite-graph sweep; `IndependentSVD` compresses each bond independently (irrepgraph_svd.jl),
# sharing only the suffix/prefix interning. Cost is `Θ(M·K) + Θ(Σ_terms span)`, linear in `N` for
# finite-range models. See `research/persistent-graph-mpo.md` for the design (§2.2's pending/started
# collision is the invariant most likely to break here).
#
# Fermionic sectors need no Jordan-Wigner strings: TensorKit's braiding carries the anticommutation
# via the odd-parity charge on the virtual bond.

using TensorKit: Sector, Vect, block, sectors, space, dim
using MatrixAlgebraKit: svd_trunc, trunctol
using SparseArrays: SparseMatrixCSC

# Invariant violation. `@noinline` keeps the cold message construction out of the hot loops, so each
# check is one never-taken branch and can be a real `throw` rather than a strippable `@assert`.
@noinline function _invariant(msg::AbstractString)
    return error("OpSum internal invariant violated: ", msg, ". This is a bug in OpSum; the reduced MPO would be wrong.")
end

# The strategies with a `_bond_basis!` method; `IndependentSVD` is not one (see `_irrep_independent_svd`).
const GraphStrategy = Union{VertexCover, SequentialSVD}

"""
    LeftVertex{I}

A left (prefix) vertex of the persistent ITO graph: it entered the current site on incoming bond
index `link` and applies the on-site ITO key `key = (op, bond, vertex)` here. There is no
fermion/JW-string slot: the odd-parity charge on the virtual bond does that bookkeeping.
"""
struct LeftVertex{I <: Sector}
    link::Int
    key::ITOKey{I}
end

"""
    ITOGraph{I}

The persistent bipartite graph over an [`ITOTermTable`](@ref), handed from one site step to the next
by `_at_site!`. A right vertex is a suffix class; classes enter at their term's first active site and
thereafter only merge. The current graph (for the bond `i-1 → i` about to be processed) is `lefts` ↔
right vertices, with per-left-vertex adjacency lists `radj`/`wadj` (right-vertex id, scalar weight);
zero-weight entries are never kept.

Fields:
* fixed suffix-class machinery: `K`, `sufid` (interned id of every contiguous column suffix, see
  [`_suffix_ids`](@ref)), the lowered exponentially decaying `channels` and `lastentry`, the last
  site at which any may still enter;
* persistent right-vertex state: `rrepr` (representative term id), the monotone cursor `rcur`/`rbond`
  behind the `O(1)` signature, and `rchan`/`rstate` for *geometric* classes (`rchan[r] == 0` an
  ordinary term class, `> 0` an automaton state of that [`ExpChannel`](@ref), `-1` the synthetic
  exhausted class a channel exits onto);
* the current graph: `lefts`, `radj`, `wadj`, `nlinks` (incoming bond dimension);
* lazy insertion (see [`_promote_pending!`](@ref)): `lazy`, `firstsite`, `pend_at`, `pendbysig`,
  `inserted`, `nremaining`, and per bond `rsent` (sentinel right-vertex id), `startleft` (its left
  vertex) and `startidx` (outgoing index of the start channel), each 0 if none;
* finish channel, the mirror image (jordanmpo.jl): `rfinish` (right vertex of the exhausted class at
  the trivial charge), `finishleft` (the left vertex its pass-through enters on), `finishidx`, each 0
  if none;
* per-site scratch, so a site step allocates nothing per bond: `slot`, `vlocal`, `firstleft` (right
  vertex → first incident left vertex, which gives its bond charge), `remap`, `siggroups`.
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

The canonical, translation-invariant name of right vertex `r` at bond `i`. An ordinary term class
(`kind = 0`) is named by the distance from the bond to its first remaining factor (`0` once
exhausted), the shape of the remaining factor list ([`_rel_suffix_ids`](@ref)) and the running bond
charge: two classes share a name exactly when they are translates of each other relative to their
bonds. A geometric class (`kind = 1`) is named by its automaton state's interned name
([`ChannelState`](@ref)); the synthetic exhausted class shares `kind = 0`'s exhausted name so it
sorts and merges with exhausted term classes. The sentinel gets the reserved name
`(-1, -1, -1, unit(I))`, which sorts first.
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
# Amortised `O(1)`: each cursor advances at most `K` times over the sweep.
#
# A geometric class has no cursor: its signature is its state's interned name in a disjoint negative id
# space (`sufid` ids are `>= 0`), so it never collides with a term's factor list or with `pendbysig`.
# The synthetic exhausted class takes the ordinary `(0, charge)` signature, so it merges with exhausted
# term classes: that merge is the done channel.
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

# Is this a *cyclic* channel state (live at every bond, forced into the cover, see `_forced_cover`)?
function _iscyclic(g::ITOGraph, r::Int)
    r <= length(g.rchan) || return false          # the sentinel
    c = g.rchan[r]
    return c > 0 && g.channels[c].states[g.rstate[r]].cyclic
end

# No remaining factors? Both a term class whose cursor ran out and the synthetic exit class are
# "exhausted" (equal signatures), which is what makes a channel's exit land on the done channel.
function _isexhausted(g::ITOGraph, r::Int)
    g.rchan[r] == -1 && return true
    return iszero(g.rchan[r]) && iszero(g.sufid[g.rcur[r], g.rrepr[r]])
end

# Append a right vertex and return its id: a term class (`repr = t`, `cur = 1`), a geometric class
# `(chan, state)` or, with `chan == -1`, the synthetic exhausted class at charge `bond`.
function _push_right!(g::ITOGraph{I}, repr::Int, cur::Int, bond::I, chan::Int, state::Int) where {I}
    push!(g.rrepr, repr)
    push!(g.rcur, cur)
    push!(g.rbond, bond)
    push!(g.rchan, chan)
    push!(g.rstate, state)
    return length(g.rrepr)
end

# The `ITOKey` term `r`'s class applies at site `i+1`, from the cursor left by `_signature!(…, i)`:
# `_op_at_ito(tt, rrepr[r], i+1)` in `O(1)`.
function _next_key(g::ITOGraph{I}, r::Int, i::Int) where {I}
    t = g.rrepr[r]
    j = g.rcur[r]
    (j <= g.K && g.tt.sites[j, t] == i + 1) && return g.tt.keys[j, t]
    return ITOKey{I}(passthrough(I), g.rbond[r], 1)
end

# First active site of each term (1 for a `K=0` identity term, which is all pass-through).
_first_sites(tt::ITOTermTable) = Int[max(1, tt.sites[1, t]) for t in 1:nterms(tt)]

"""
    ITOGraph(tt::ITOTermTable{I}, N; lazy = true, channels = ExpChannel{I}[]) -> ITOGraph{I}

Seed the persistent graph: intern the column suffixes, create the right vertices, and bucket the
initial left vertices by the site-1 key against the single left-boundary link. `Θ(M·K)`.

With `lazy = true` (the default) only terms *active at site 1* get a right vertex; the rest are
represented by a sentinel right vertex on the identity/start channel and inserted when reachable
(`_promote_pending!` / `_build_next_graph!`), which makes the sweep cost `Θ(Σ_terms span)` instead of
`Θ(N·M)`. `lazy = false` seeds every term eagerly (used by `SequentialSVD`, whose dense SVD dominates
anyway). `jordan = true` forces the finish class into every cover (`_force_finish!`), as `mpo` asks
on a finite chain without decaying channels.

`channels` are lowered exponentially decaying interactions ([`_lower_channels`](@ref)): not terms,
they enter on the start channel at every site matching their phase and are thereafter ordinary
(cyclic) right vertices. They require `lazy = true`.
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

    # channels anchored at site 1 enter here, on the same start channel a term does
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

    # pending terms: the site each enters at, and its pre-start signature (full factor list, trivial
    # charge) so a colliding class can promote it
    pend_at = [Int[] for _ in 1:N]
    pendbysig = Dictionary{Tuple{Int, I}, Int}()
    nremaining = 0
    for t in 1:M
        inserted[t] && continue
        nremaining += 1
        push!(pend_at[firstsite[t]], t)
        insert!(pendbysig, (sufid[1, t], unit(I)), t)   # injective: `sufid[1, t]` fixes the term
    end

    # the start channel's left vertex, where the sentinel and every injected term or channel entry
    # hangs off (a `K=0` term shares its `(link, key)`, so the bucket may already exist)
    startleft = (nremaining > 0 || 1 < lastent) ? bucket!(ITOKey{I}(passthrough(I), unit(I), 1)) : 0

    cap = length(rrepr) + 1   # right-vertex ids are at most one sentinel beyond the live ones
    return ITOGraph{I}(
        tt, N, arity(tt), sufid, relsufid, channels, lastent,
        rrepr, rcur, rbond, rchan, rstate, lefts, radj, wadj, 1,
        lazy, firstsite, pend_at, pendbysig, inserted, nremaining, 0, startleft, 0, 0, 0, 0, jordan,
        zeros(Int, cap), zeros(Int, cap), zeros(Int, cap), zeros(Int, cap),
        Dictionary{Tuple{Int, I}, Int}()
    )
end

# Phase 1: suffix-merge the right vertices, i.e. group those "equal from site i+1 on" by their
# `_signature!`. `Θ(live)`. Returns `remap[old_right_id] -> new_right_id` and replaces the `g.r*` state.
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

# Apply a right-vertex remap (or none) to left vertex `lv`'s adjacency, summing weights of edges that
# land on the same right vertex. In place via the `slot` scratch table: `Θ(deg)`, no allocation.
# The surviving order is first-encounter; `_canonicalise_rights!` sorts it later.
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
    # drop edges whose weight cancelled to zero: they must not reach the cover (a wasted bond index)
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

# `slot` is all-zero between merges, so only newly grown space needs clearing.
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

`_op_at_ito` fills idle sites with a pass-through carrying the **running** bond charge, so a started
term whose accumulated charge has fused back to `unit(I)` is indistinguishable, over its idle sites,
from a term that has not started. If its remaining factors then coincide with a pending term's whole
factor list, the two suffix classes are genuinely equal and the eager sweep merges them (covering the
shared right vertex instead of spending a bond index). That merge has to be reproduced at the
*earliest* bond where it applies; suffix-equality-from-`i+1` is monotone in `i`, so probing every bond
finds it exactly once. Cost: one hash probe per live right vertex (`pendbysig` is injective).
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
    # the start channel may already have an edge to this class, so sum parallel edges
    promoted && _merge_edges!(g, lv, nothing)
    return g
end

"""
    _canonicalise_rights!(g, i)

Renumber the live right vertices into canonical order (ascending [`_rdesc`](@ref)) and sort every
adjacency list to match.

This makes the sweep a deterministic function of the *canonical* graph, not of its construction
history. Hopcroft–Karp and König depend on the adjacency scan order, so without it isomorphic bonds
can answer differently. Invisible on a finite chain (any minimum cover will do); fatal on a periodic
lattice, where the bond basis converges only up to permutation and the extracted unit cell does not
close. Components come out in first-left-vertex order with ascending ids, so canonical ids here plus
canonical index order in `_at_site!` pin the whole sweep. Called before the sentinel is attached.
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

# Phases 1 & 2, shared by every graph-sweep strategy: suffix-merge, apply the remap, promote pending
# terms whose class just became live, canonicalise, attach the sentinel standing in for the
# still-pending terms, and record `g.firstleft` while checking every right vertex is pure in the
# incoming bond charge (block-diagonality, one comparison per edge). The sentinel is not a term: it
# takes the id one past the real right vertices, so it is skipped by everything iterating `g.rrepr`
# and discarded each bond. Returns `(nU, nV)`.
function _prepare_bond!(g::ITOGraph{I}, i::Int) where {I}
    g.rsent = 0
    remap = _suffix_merge!(g, i)
    nV = length(g.rrepr)
    _apply_remap!(g, remap, nV)
    _promote_pending!(g, i)
    _canonicalise_rights!(g, i)

    # the finish class: exhausted at the trivial charge (unique after the merge). `_isexhausted`, not
    # a bare `sufid` lookup: channel right vertices have no representative term.
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
                    _invariant("bond index not sector-pure (block-diagonality violated)")
            end
        end
    end

    return nU, nV
end

# Phase 5: build the next graph on the SAME (persistent) right vertices, with fresh left vertices
# tagged by the outgoing bond index `j` as `link` and bucketed by the right vertex's next-site key.
# `nextedges_global[j]` lists the `(right_vertex, weight)` edges outgoing index `j` forwards. Mutates
# `g` into the graph for bond `i → i+1`. The next-site key depends only on the right vertex, so it is
# computed once per right vertex.
#
# Lazy insertion *injects* here: a term whose first active site is `i+1` gets its right vertex,
# hanging off the start channel with the term's coefficient as edge weight (what the eager sweep would
# have carried along that channel since bond 0). A decaying channel enters the same way, at *every*
# site matching its phase.
#
# A geometric right vertex is the one class with several successors: its automaton state has a
# transition per string letter (continue, weight `λ·c`) and, at `δ = 1`, one for the exit factor. Each
# becomes its own left-vertex bucket; the continuation puts `λ · pass-through` on the channel's
# diagonal and the exit reaches the done channel. Transitions that could no longer complete inside
# `1:N` are dropped, which keeps the last bond of a finite chain one-dimensional.
function _build_next_graph!(
        g::ITOGraph{I}, i::Int, nout::Int,
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}}
    ) where {I}
    # right for term classes and the synthetic exhausted class (bare pass-through at the running
    # charge); live channel states are handled below
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

    # automaton successor states and the exit class may not exist yet: created here on demand
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
        # `nremaining > 0` (or an entry still to come) means the sentinel existed at this bond, which
        # forces a start channel; otherwise `pend_at[i+1]` and the channel loop are no-ops.
        ((iszero(g.nremaining) && i + 1 > g.lastentry) || !iszero(startidx)) ||
            _invariant("terms remain to the right of site $i with no start channel to enter on")
        for t in g.pend_at[i + 1]
            g.inserted[t] && continue         # already promoted into a colliding class
            g.inserted[t] = true
            g.nremaining -= 1
            b = bucket!(startidx, g.tt.keys[1, t])
            push!(next_radj[b], _push_right!(g, t, 1, unit(I), 0, 0))
            push!(next_wadj[b], g.tt.coeffs[t])
        end
        # every channel whose phase matches site `i+1` enters there, on the start channel
        for (ci, c) in enumerate(g.channels)
            isentry(c, i + 1, g.N) || continue
            b = bucket!(startidx, c.entrykey)
            st = c.states[c.start]
            push!(next_radj[b], geomvertex!(ci, c.start, st.bond))
            push!(next_wadj[b], c.coeff)
        end
        # the start channel must survive even when it forwards nothing: the next sentinel hangs off it
        g.startleft = (g.nremaining > 0 || i + 1 < g.lastentry) ?
            bucket!(startidx, ITOKey{I}(passthrough(I), unit(I), 1)) : 0
    end

    # the finish channel forwards one edge, to the exhausted class (key: the trivial pass-through), so
    # its continuation is a single left vertex; the next bond recognises its finish index by it
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

# One site step, five phases: (1) suffix-merge the right vertices; (2) connected components / bond
# assembly; (3) the strategy's bond-basis choice; (4) assemble the bond; (5) build the next graph.
# Phases 1, 2 and 5 are strategy-independent; `_bond_basis!` covers 3 & 4. Returns
# `(Ws_i, secW_i, startidx, finishidx)` and mutates `g` into the graph for bond `i → i+1`.
function _at_site!(g::ITOGraph{I}, i::Int, strategy::GraphStrategy = VertexCover()) where {I}
    nU, nV = _prepare_bond!(g, i)
    g.startidx = 0
    g.finishidx = 0
    nout, site_dict, secW, nextedges_global = _bond_basis!(g, i, nU, nV, strategy)
    Ws_i = sparse_from_dict(site_dict, (g.nlinks, nout))
    i < g.N && _build_next_graph!(g, i, nout, nextedges_global)
    return Ws_i, secW, g.startidx, g.finishidx
end

# Lazy insertion pays off only with the sparse adjacency; the SVD strategy builds a dense matrix anyway.
_graph_lazy(::VertexCover) = true
_graph_lazy(::SequentialSVD) = false

# Resolve `trunc === nothing` (the lossless default) once per sweep rather than once per bond.
_resolve_trunc(s::BondStrategy) = s

"""
    _irrep_graph_channels(tt::ITOTermTable{I}, N, strategy, jordan; channels) -> (Ws, bondsectors, starts, finishes)

The persistent-graph sweep with a bond-basis `strategy` ([`VertexCover`](@ref) or
[`SequentialSVD`](@ref)). `Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}` and
`bondsectors::Vector{Vector{I}}` are what `mpo_terms` / `irrep_mpo_tensors` consume; `starts[i]` /
`finishes[i]` are the bond indices of the start ("nothing placed yet") and finish ("everything
placed") channels at the bond right of site `i`, or `0` where the cover spent no index on them (always
for `SequentialSVD`, whose bond basis mixes prefix states). `channels` are lowered exponentially
decaying interactions ([`_lower_channels`](@ref)), which live beside the term table; a model may
consist of nothing else.

`jordan = true` also forces the finish class into every cover ([`_force_finish!`](@ref)), which can
change the bond basis away from the unconstrained minimum (rarely: a constant term next to two-site
terms costs one index at the first bond, pinned in `test/test_irrep_graph.jl`).
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
    _irrep_sweep(tt::ITOTermTable{I}, N, strategy::BondStrategy; channels) -> (Ws, bondsectors)

The reduced-MPO compression of `tt` over `N` sites, without identity channels (the entry point `mpo`
dispatches to); the strategy decides between the persistent-graph sweep and the independent pass.
"""
function _irrep_sweep(
        tt::ITOTermTable{I}, N::Int, strategy::GraphStrategy;
        channels::Vector{ExpChannel{I}} = ExpChannel{I}[]
    ) where {I}
    Ws, bondsectors, _, _ = _irrep_graph_channels(tt, N, strategy, false; channels)
    return (Ws, bondsectors)
end

"""
    _irrep_channels(tt::ITOTermTable, N, strategy::BondStrategy) -> (Ws, bondsectors, starts, finishes)

[`_irrep_sweep`](@ref) with the identity-channel indices of [`_irrep_graph_channels`](@ref) and the
finish class forced into the cover. The independent-SVD pass reports none, so Jordan emission pads
every channel.
"""
_irrep_channels(tt::ITOTermTable, N::Int, s::GraphStrategy) = _irrep_graph_channels(tt, N, s, true)
