# Reduced-MPO sweeps for the ITO automaton
# ========================================
# One sweep skeleton with a pluggable bond-basis strategy (algorithms.jl), plus the one strategy that
# cannot share it.
#
# * `VertexCover` (default) and `SequentialSVD` run the *persistent bipartite graph* sweep
#   `_irrep_graph_sweep` — a port of ITensorMPOConstruction.jl's persistent-graph + `at_site!`
#   architecture onto OpSum's non-abelian (TensorKit `Sector`) ITO machinery. An explicit bipartite
#   graph is handed from one site step to the next instead of re-materialising every strand's suffix
#   path each bond. Four of the five site-step phases are shared; only `_bond_basis!` differs.
# * `IndependentSVD` compresses every bond on the *raw* prefix/suffix classes, independently of its
#   neighbours, so it cannot ride the persistent graph — it gets its own pass
#   (`_irrep_independent_svd`), but shares the class-interning machinery (`_prefix_ids`/`_suffix_ids`)
#   rather than duplicating it.
#
# Cost of the graph sweep is `Θ(M·K)` to intern the suffix classes plus `Θ(Σ_terms span)` for the
# sweep, i.e. linear in N for a finite-range model. Two things buy that, and both are load-bearing:
# suffix classes are named by an interned `O(1)` signature rather than a materialised path
# (`_suffix_ids`, `_signature!`), and a term's right vertex is created only once it is reachable
# (`_promote_pending!` and the injection in `_build_next_graph!`). See
# research/persistent-graph-mpo.md §2 — in particular §2.2, whose pending-versus-started class
# collision is the invariant a change here is most likely to break.
#
# INVARIANT CHECKS. The sector-purity / cover-validity / start-channel checks below guard *silently
# wrong output*, not crashes: a violated one means the emitted MPO represents a different operator.
# They are therefore explicit `throw`s (via the `@noinline _invariant` helper, so the message is only
# built on failure and the check itself costs one comparison), never `@assert`, which `--check-bounds`
# / `-O3 --inline` builds are entitled to elide.
#
# Non-abelian mapping of the persistent graph (see research/itensor-mpograph-construction.md §9):
# * Right vertices  = suffix classes of the `ITOTermTable`, entering at their term's first active site
#   and thereafter only merging. ITensor's "term + additive QN flux" becomes "term + running fusion
#   charge", carried in each site's `ITOKey.bond` (a fusion *outcome*, not a sum).
# * Left vertices   = `LeftVertex(link, key::ITOKey)` — the incoming bond index plus the on-site ITO
#   key applied here. Rebuilt each site.
# * The graph sweep stays SCALAR: reduced coefficients are `ComplexF64`, min-vertex-cover runs on the
#   scalar adjacency. Non-abelian structure enters only (a) in what makes a bond state distinct (the
#   augmented `ITOKey` → `bondsectors`) and (b) later, at tensor assembly (`irrep_mpo_tensors`).
# * Connected components are pure in the outgoing bond charge (`ITOKey.bond`) — checked, see above.
#
# Supported arity K ≥ 0 (identity, on-site field, and caterpillar coupling of any number of sites;
# the suite covers K ≤ 3 and the examples K = 4). The `ITOKey.vertex` multiplicity label is threaded
# through `LeftVertex.key` but is `1` throughout the multiplicity-free scope, so the reduced `(Ws,
# bondsectors)` contract is unchanged. `GenericFusion` multi-channel coupling is out of scope.
#
# Fermionic (graded) sectors ARE supported: TensorKit's braiding carries the anticommutation through
# the ITO algebra and the odd-parity charge flowing along the virtual bond does the bookkeeping, so
# no Jordan–Wigner strings appear. That is why ITensor's fermion/JW slot on `LeftVertex` is omitted —
# it is subsumed by the sector structure, not missing.

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
  *contiguous column suffix* of `tt` — see [`_suffix_ids`](@ref);
* persistent right-vertex state (shrinks via the suffix-merge): `rrepr` (representative term id) plus
  the monotone cursor `rcur`/`rbond` that turns "the suffix path from site `i+1`" into an `O(1)`
  two-word signature;
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

"""
    _prefix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Intern every term's *contiguous column prefixes* — the mirror image of [`_suffix_ids`](@ref).
`preid[j, t]` is a dense integer id for the factor list `1:j-1` of term `t` (so `preid[1, t] == 0`,
the empty prefix), built top-down in `Θ(M·K)`.

Equality of ids **is** equality of the prefix path `o_t[1:b]`, with no charge component needed: at
every idle site of the prefix `_op_at_ito` fills in a pass-through whose running charge is fixed by
the factors to its left, i.e. by the factor list itself. (The suffix is the asymmetric one — the
idle sites *before* its first remaining factor carry the charge accumulated by the prefix, which is
why `_signature!` pairs `sufid` with the running charge.)

Used by the `IndependentSVD` sweep, which classifies both sides of every bond from scratch.
"""
function _prefix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    preid = zeros(Int, K + 1, M)      # row 1 stays 0 == the empty prefix
    intern = Dictionary{Tuple{Int, Int, ITOKey{I}}, Int}()
    nid = 0
    for t in 1:M
        for j in 1:K
            s = tt.sites[j, t]
            iszero(s) && break        # padding: no further factors
            trans = (preid[j, t], s, tt.keys[j, t])
            id = get(intern, trans, 0)
            if iszero(id)
                nid += 1
                id = nid
                insert!(intern, trans, id)
            end
            preid[j + 1, t] = id
        end
    end
    return preid
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
which is the form the `SequentialSVD` strategy uses (its per-bond dense SVD dominates anyway).

`jordan = true` additionally forces the finish class into every cover (`_force_finish!`), which is
what Jordan emission needs and what `irrep_mpo` deliberately does not do.
"""
function ITOGraph(tt::ITOTermTable{I}, N::Int; lazy::Bool = true, jordan::Bool = false) where {I}
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

"""
    _force_finish!(cUbits, cVbits, localadj, pf) -> Bool

Move the finish class (component-local right index `pf`) into the cover, and drop every left vertex
that this makes redundant. Returns whether anything changed.

Why it is needed. A minimum cover is free to cover the finish class from the *left* instead, and does
so exactly when it can: if the finish class and one incident left vertex form an isolated matched pair
— what a bond at which nothing new finishes looks like — König's alternating search visits neither, so
the left vertex is the one that lands in the cover. The resulting bond index still means "already
finished", but it is the covered-left index of a term's last factor, so it emits that factor's
*letter* weighted by the term coefficient, not the identity.
Jordan form needs a channel that emits exactly `1 · id` from the previous bond's finish index, and
that is only the covered-*right* reading. Since a covered-left finish then has to be padded around,
forcing is never a loss: it grows the cover by at most one (a cover cannot shrink below the minimum,
so at most one left vertex is dropped), and it saves exactly the one padded index it would otherwise
have cost. Dropping a left vertex whose neighbours are all covered is safe — a left vertex is only
needed for the edges its neighbours do not cover.

The start channel needs no counterpart: §2.2 of research/persistent-graph-mpo.md shows König never
covers a degree-one right vertex, so the sentinel's left vertex `L₀` is always covered-left already.
`L₀` is also never adjacent to the finish class — every class it carries has at least one factor left
to place — so forcing cannot remove it.
"""
function _force_finish!(cUbits, cVbits, localadj::Vector{Vector{Int}}, pf::Int)
    cVbits[pf] && return false
    cVbits[pf] = true
    @inbounds for k in eachindex(localadj)
        cUbits[k] || continue
        all(q -> cVbits[q], localadj[k]) && (cUbits[k] = false)
    end
    return true
end

# Per-component minimum-vertex-cover backend (the VC path). Given a connected component `(us, vs)`
# (global left/right vertex ids), it chooses the component's bond basis via
# `min_vertex_cover_bipartite` and returns, for that component: `rank`, `blocks`
# (`(incoming_link, local_bond_index, localop)`), `nextedges` (per local bond index, the
# `(right_vertex_id, weight)` edges to forward; empty at the last site), `secs` (the bond charge per
# local index), `startidx` (the local index of the identity/start channel, 0 if this component does
# not hold it) and `finishidx` (likewise for the finish channel). The covered-U / covered-V
# coefficient-flow is ITensor's (doc §6): covered-left forwards its edge weights unchanged and emits
# the bare letter; covered-right resets the forwarded weight to 1 and folds `key.op × weight` into
# the block for every uncovered incident left.
#
# THE FINISH CHANNEL. Jordan form (jordanmpo.jl) needs a bond index meaning "every factor is placed",
# reachable *only* from the previous bond's finish index, so that the emitted matrix has no entry
# below-left of the `(end, end)` corner. Exactly two vertices can carry that meaning: the right vertex
# `g.rfinish` (the exhausted, trivial-charge suffix class) and the left vertex `g.finishleft =
# (previous finish index, pass-through)`, whose *only* neighbour is `g.rfinish`. On the Jordan path
# (`g.jordan`) `_force_finish!` has already put `g.rfinish` in the cover, so `g.finishleft` is then
# uncovered and folds `passthrough × 1` in; without forcing, whichever of the two the cover takes is
# read here, and never both (covering both would leave the degree-1 left vertex redundant, which a
# minimum cover has not got). Either reading emits the bare pass-through with weight 1, so the
# `(end, end)` corner is an exact identity.
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
    LOp = SiteOperator{I}
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
    if g.jordan && !iszero(g.rfinish)
        pf = findfirst(==(g.rfinish), vs)     # `Θ(|vs|)`, i.e. `Θ(nV)` summed over the components
        pf === nothing || _force_finish!(cUbits, cVbits, localadj, pf)
    end
    cU = findall(cUbits)
    cV = findall(cVbits)
    nleft = length(cU)
    rank = nleft + length(cV)

    blocks = Tuple{Int, Int, LOp}[]
    nextedges = [Tuple{Int, ComplexF64}[] for _ in 1:rank]
    secs = Vector{I}(undef, rank)
    startidx = 0
    finishidx = 0

    # covered-left vertices → local bond indices 1 … nleft ("a term starts its operator here")
    for (m, lu) in enumerate(cU)
        iu = us[lu]
        lv = g.lefts[iu]
        secs[m] = lv.key.bond
        iu == g.startleft && (startidx = m)
        iu == g.finishleft && (finishidx = m)
        if i == N
            iszero(g.rsent) || _invariant("the sentinel must be gone by the last site")
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
        if iv == g.rfinish
            iszero(finishidx) ||
                _invariant("finish channel covered on both sides (the cover is not minimum)")
            finishidx = m
        end
        if iv == g.rsent
            iszero(startidx) || _invariant("start channel covered on both sides")
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
            iszero(m) && _invariant("edge left uncovered by the minimum vertex cover")
            push!(blocks, (lv.link, m, lv.key.op * w))
        end
    end

    return rank, blocks, nextedges, secs, startidx, finishidx
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

    # the finish class: every factor placed (exhausted suffix) at the trivial running charge. The
    # suffix merge just made signatures unique, so there is at most one such right vertex.
    g.rfinish = 0
    @inbounds for r in eachindex(g.rrepr)
        if iszero(g.sufid[g.rcur[r], g.rrepr[r]]) && g.rbond[r] == unit(I)
            g.rfinish = r
            break
        end
    end

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
                g.lefts[firstleft[rid]].key.bond == bond ||
                    _invariant("bond index not sector-pure (block-diagonality violated)")
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
        (iszero(g.nremaining) || !iszero(startidx)) ||
            _invariant("terms remain to the right of site $i with no start channel to enter on")
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

# Phases 3 & 4 for [`VertexCover`](@ref): split the bond into connected components, run the
# per-component minimum vertex cover (`_vc_component`), and concatenate the component ranks into one
# bond (offsets), collecting the per-index charges and the forwarded edges. A minimum vertex cover of
# a disjoint union is the union of the components' minimum covers (König per component), so this is a
# pure decomposition — same bond dimension, smaller matching problems.
#
# Returns the `_bond_basis!` contract `(nout, site_dict, secW, nextedges_global)`.
function _bond_basis!(g::ITOGraph{I}, i::Int, nU::Int, nV::Int, ::VertexCover) where {I}
    LOp = SiteOperator{I}
    us_of_comp, vs_of_comp = bipartite_connected_components(g.radj, nV)

    secW = I[]
    nextedges_global = Vector{Tuple{Int, ComplexF64}}[]
    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    offset = 0
    for (us, vs) in zip(us_of_comp, vs_of_comp)
        rank, blocks, nextedges, secs, startidx, finishidx = _vc_component(g, us, vs, i)
        for (link, m, op) in blocks
            increaseindex!(site_dict, CartesianIndex(link, offset + m), op)
        end
        for m in 1:rank
            push!(secW, secs[m])
            push!(nextedges_global, nextedges[m])
        end
        if !iszero(startidx)
            iszero(g.startidx) ||
                _invariant("the start channel appeared in more than one component")
            g.startidx = offset + startidx
        end
        if !iszero(finishidx)
            iszero(g.finishidx) ||
                _invariant("the finish channel appeared in more than one component")
            g.finishidx = offset + finishidx
        end
        offset += rank
    end
    (iszero(g.startidx) || g.startidx != g.finishidx) ||
        _invariant("the start and finish channels resolved to the same bond index")
    return offset, site_dict, secW, nextedges_global
end

# Phases 3 & 4 for [`SequentialSVD`](@ref) (ITensor's `at_site!` with the QR/SVD backend, doc §6 "The
# QR backend"). Instead of a per-component minimum vertex cover, the WHOLE bond's scalar coefficient
# matrix is assembled as a charge-graded `TensorMap C : Ppre ← Psuf` (block-diagonal in the bond
# charge, so `svd_trunc` does the per-sector SVD *and* the global-across-sectors truncation at once,
# respecting quantum dimensions). Keeping `U` (left singular vectors) as the compressed bond basis:
# each outgoing bond index `m` is a linear combination `U[u, m]` of prefix states, emitting
# `key.op * U[u, m]` into the `(link, m)` block; the residual `R = S·Vᴴ` forwards the coefficient onto
# the next bond's edges (folded into the block at the last site).
#
# This is where the two SVD semantics part: the basis handed to the next bond is `U`, i.e. whatever
# survived truncation here — see [`SequentialSVD`](@ref) versus [`IndependentSVD`](@ref).
function _bond_basis!(g::ITOGraph{I}, i::Int, nU::Int, nV::Int, strategy::SequentialSVD) where {I}
    LOp = SiteOperator{I}
    N = g.N
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
        all(uCharge[u] == q for u in conn) ||
            _invariant("bond index not sector-pure (block-diagonality violated)")
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

    U, S, Vt = svd_trunc(C; trunc = strategy.trunc)
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

    return r, site_dict, secW, nextedges_global
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
_resolve_trunc(s::SequentialSVD) = SequentialSVD(something(s.trunc, trunctol(rtol = eps(Float64))))

"""
    _irrep_graph_sweep(tt::ITOTermTable{I}, N, strategy) -> (Ws, bondsectors)

The persistent-graph reduced-MPO sweep, run with a bond-basis `strategy` ([`VertexCover`](@ref) or
[`SequentialSVD`](@ref)). Produces the `(Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}},
bondsectors::Vector{Vector{I}})` contract that `mpo_terms` / `irrep_mpo_tensors` consume.
"""
function _irrep_graph_sweep(tt::ITOTermTable, N::Int, strategy::GraphStrategy)
    Ws, bondsectors, _, _ = _irrep_graph_channels(tt, N, strategy, false)
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
        tt::ITOTermTable{I}, N::Int, strategy::GraphStrategy, jordan::Bool
    ) where {I}
    LOp = SiteOperator{I}
    nterms(tt) == 0 && return (SparseMatrixCSC{LOp, Int}[], Vector{I}[], Int[], Int[])

    strategy = _resolve_trunc(strategy)
    g = ITOGraph(tt, N; lazy = _graph_lazy(strategy), jordan)
    Ws = Vector{SparseMatrixCSC{LOp, Int}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    starts = zeros(Int, N)
    finishes = zeros(Int, N)
    for i in 1:N
        Ws[i], bondsectors[i], starts[i], finishes[i] = _at_site!(g, i, strategy)
    end
    return (Ws, bondsectors, starts, finishes)
end


# Independent-SVD sweep — not a graph sweep
# =========================================
# `IndependentSVD` compresses every bond on the *raw* prefix/suffix classes of the term table,
# independently of what its neighbours kept, so it cannot ride the persistent graph (whose whole
# point is that bond `b` is expressed in the basis bond `b-1` left behind). It therefore gets its own
# pass — but it shares the class-interning machinery above rather than duplicating it: prefix classes
# are `_prefix_ids`, suffix classes the `(_suffix_ids, running bond charge)` signature of §2.1.

"""
    _irrep_independent_svd(tt::ITOTermTable{I}, N, trunc) -> (Ws, bondsectors)

Per-bond-*independent* SVD reduced-MPO sweep: one coefficient matrix per bond over the raw
prefix/suffix classes, keep the left singular vectors as that bond's compressed basis, then project
the vertex operators into the compressed bases (`W_op = U_{i-1}' · C_op · U_i` per ITO letter).

The ITO-specific part: each bond's coefficient matrix is a *charge-graded* `TensorMap C_b : Ppre ←
Psuf`, both spaces graded by the running bond charge, so `C_b` is block-diagonal in that charge and
`svd_trunc` does the per-sector SVD *and* the global-across-sectors truncation at once (respecting
the quantum dimensions); the retained bond space gives `bondsectors` directly. The compression acts
on the symbolic bond coefficients only — entries stay ITO letters times scalars — so the output
`(Ws, bondsectors)` feeds `irrep_mpo_tensors` unchanged.

`trunc === nothing` ⇒ lossless default, and then this agrees with [`SequentialSVD`](@ref) on the
internal bonds. Under truncation they differ by design; see [`SVDBondAlgorithm`](@ref).

Classes are named, not materialised: the prefix class at bond `b` is the interned prefix factor list
(`_prefix_ids` — the pass-through fill of every idle site to its left is fixed by that list), and the
suffix class is the two-word signature `(sufid, running charge)` the graph sweep uses. Only the
current bond's `Θ(M)` class assignment is held at a time, plus one `interned id → dense column`
dictionary per bond (`Θ(Σ_b n_pre(b))`, i.e. `Θ(N)` for a finite-range model) so that phase 3 can
address the same columns phase 2 built.
"""
function _irrep_independent_svd(tt::ITOTermTable{I}, N::Int, trunc) where {I}
    T = ComplexF64
    Op = ITOKey{I}
    LOp = SiteOperator{I}
    M = nterms(tt)
    M == 0 && return (SparseMatrixCSC{LOp, Int}[], Vector{I}[])

    K = arity(tt)
    preid = _prefix_ids(tt)
    sufid = _suffix_ids(tt)
    truncstrat = something(trunc, trunctol(rtol = eps(Float64)))
    nb = max(N - 1, 0)

    # --- 1. Per internal bond: classify both sides, SVD the charge-graded matrix, keep U -------
    #   `bond_Us[b]` is the (n_pre × r_b) left isometry as a plain matrix (block-diagonal in the
    #   charge, columns grouped per sector); `bond_secs[b]` is the retained charge per column;
    #   `predense[b]` maps an interned prefix id to its dense column, so phase 3 can re-derive the
    #   same numbering without storing an `M × (N-1)` id matrix.
    bond_Us = Vector{Matrix{T}}(undef, nb)
    bond_secs = Vector{Vector{I}}(undef, nb)
    predense = [Dictionary{Int, Int}() for _ in 1:nb]

    cursor = zeros(Int, M)          # term -> #active factors at sites <= b (monotone in b)
    pterm = zeros(Int, M)           # term -> dense prefix column at the current bond
    sterm = zeros(Int, M)           # term -> dense suffix column at the current bond
    preQ, sufQ = I[], I[]           # dense class -> bond charge
    pre_deg, suf_deg = Int[], Int[] # dense class -> degeneracy index within its charge sector
    pre_mult, suf_mult = Dict{I, Int}(), Dict{I, Int}()
    sufdense = Dictionary{Tuple{Int, I}, Int}()

    for b in 1:nb
        pd = predense[b]
        empty!(sufdense)
        empty!(preQ)
        empty!(sufQ)
        empty!(pre_deg)
        empty!(suf_deg)
        empty!(pre_mult)
        empty!(suf_mult)

        for t in 1:M
            j = cursor[t]
            @inbounds while j < K && !iszero(tt.sites[j + 1, t]) && tt.sites[j + 1, t] <= b
                j += 1
            end
            cursor[t] = j
            q = iszero(j) ? unit(I) : tt.keys[j, t].bond

            p = get(pd, preid[j + 1, t], 0)
            if iszero(p)
                push!(preQ, q)
                pre_mult[q] = get(pre_mult, q, 0) + 1
                push!(pre_deg, pre_mult[q])
                p = length(preQ)
                insert!(pd, preid[j + 1, t], p)
            elseif preQ[p] != q
                # unreachable: the prefix factor list fixes the running charge. Kept because a wrong
                # class here silently mixes charge sectors into one bond index.
                _invariant("prefix class not sector-pure")
            end
            pterm[t] = p

            sig = (sufid[j + 1, t], q)   # the charge is part of the key, so purity is structural
            s = get(sufdense, sig, 0)
            if iszero(s)
                push!(sufQ, q)
                suf_mult[q] = get(suf_mult, q, 0) + 1
                push!(suf_deg, suf_mult[q])
                s = length(sufQ)
                insert!(sufdense, sig, s)
            end
            sterm[t] = s
        end

        C = zeros(T, Vect[I](pre_mult) ← Vect[I](suf_mult))
        for t in 1:M
            p, s = pterm[t], sterm[t]
            block(C, preQ[p])[pre_deg[p], suf_deg[s]] += tt.coeffs[t]
        end

        U, S, _ = svd_trunc(C; trunc = truncstrat)
        Wb = space(S, 1)   # retained bond space (⊕ charge sectors with truncated multiplicities)

        # flatten U into an (npre × r_b) block-diagonal matrix, columns grouped per sector
        npre = length(preQ)
        r_b = sum(q -> dim(Wb, q), sectors(Wb); init = 0)
        Umat = zeros(T, npre, r_b)
        secs = I[]
        col = 0
        for q in sectors(Wb)
            Ub = block(U, q)              # (pre_mult[q] × Wb_mult[q])
            for dcol in 1:size(Ub, 2)
                col += 1
                push!(secs, q)
                for p in 1:npre
                    preQ[p] == q || continue
                    Umat[p, col] = Ub[pre_deg[p], dcol]
                end
            end
        end
        col == r_b || _invariant("retained bond space does not match its per-sector dimensions")
        bond_Us[b] = Umat
        bond_secs[b] = secs
    end

    # --- 2. Project each vertex operator into the compressed bond bases -----------------------
    #   W_op = U_{i-1}' · C_op · U_i, per ITO letter, in the uncompressed (pre_{i-1}, pre_i) basis
    #   (boundary bonds are the 1×1 identity). Emits the letter times the compressed coefficient.
    r = [size(bond_Us[b], 2) for b in 1:nb]
    sizes = Tuple{Int, Int}[(b == 1 ? 1 : r[b - 1], b == N ? 1 : r[b]) for b in 1:N]
    dicts = [Dictionary{CartesianIndex{2}, LOp}() for _ in 1:N]
    fill!(cursor, 0)                # term -> #active factors at sites <= i-1, walked forward again
    for i in 1:N
        U_left = i > 1 ? bond_Us[i - 1] : ones(T, 1, 1)
        U_right = i < N ? bond_Us[i] : ones(T, 1, 1)
        nL, nR = size(U_left, 1), size(U_right, 1)

        op_coeffs = Dictionary{Op, Matrix{T}}()
        for t in 1:M
            jprev = cursor[t]
            active = jprev < K && tt.sites[jprev + 1, t] == i
            jcur = jprev + (active ? 1 : 0)
            cursor[t] = jcur
            key = active ? tt.keys[jcur, t] :
                ITOKey{I}(passthrough(I), iszero(jprev) ? unit(I) : tt.keys[jprev, t].bond, 1)
            j = i > 1 ? predense[i - 1][preid[jprev + 1, t]] : 1
            l = i < N ? predense[i][preid[jcur + 1, t]] : 1
            Cmat = get!(() -> zeros(T, nL, nR), op_coeffs, key)
            if i == N
                Cmat[j, l] += tt.coeffs[t]   # accumulate: many terms can share the same prefix
            else
                Cmat[j, l] = one(T)          # deterministic: same (j, l, key) ⇒ same successor
            end
        end

        for (key, C_op) in pairs(op_coeffs)
            W_op = U_left' * C_op * U_right
            lop = convert(LOp, key.op)
            for col in 1:size(W_op, 2), row in 1:size(W_op, 1)
                iszero(W_op[row, col]) && continue
                increaseindex!(dicts[i], CartesianIndex(row, col), lop * W_op[row, col])
            end
        end
    end

    # bond to the right of site i: internal bonds from the SVD, the right boundary is trivial
    bondsectors = Vector{I}[i < N ? bond_secs[i] : I[unit(I)] for i in 1:N]
    return (map(sparse_from_dict, dicts, sizes), bondsectors)
end

"""
    _irrep_sweep(tt::ITOTermTable{I}, N, strategy::BondStrategy) -> (Ws, bondsectors)

Run the reduced-MPO compression of `tt` over `N` sites with the given bond-basis strategy. This is
the single entry point `irrep_mpo` (irrepmpo.jl) dispatches to; the strategy decides whether that is
the persistent-graph sweep or the independent per-bond pass.
"""
_irrep_sweep(tt::ITOTermTable, N::Int, strategy::GraphStrategy) = _irrep_graph_sweep(tt, N, strategy)
function _irrep_sweep(tt::ITOTermTable, N::Int, strategy::IndependentSVD)
    return _irrep_independent_svd(tt, N, strategy.trunc)
end

"""
    _irrep_channels(tt::ITOTermTable, N, strategy::BondStrategy) -> (Ws, bondsectors, starts, finishes)

[`_irrep_sweep`](@ref) with the identity-channel indices [`_irrep_graph_channels`](@ref) documents,
and with the finish class forced into the cover. The independent-SVD pass has no persistent bond
identity at all, so it reports none and Jordan emission pads every channel.
"""
_irrep_channels(tt::ITOTermTable, N::Int, s::GraphStrategy) = _irrep_graph_channels(tt, N, s, true)
function _irrep_channels(tt::ITOTermTable, N::Int, strategy::IndependentSVD)
    Ws, bondsectors = _irrep_independent_svd(tt, N, strategy.trunc)
    return (Ws, bondsectors, zeros(Int, length(Ws)), zeros(Int, length(Ws)))
end
