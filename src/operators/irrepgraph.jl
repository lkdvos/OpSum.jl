# Persistent bipartite-graph MPO construction for the ITO automaton
# =================================================================
# A port of ITensorMPOConstruction.jl's persistent-graph + `at_site!` sweep architecture onto
# OpSum's non-abelian (TensorKit `Sector`) ITO machinery. It builds the *same* reduced MPO as the
# transient-frontier sweep `_irrep_bipartite` (irreptermtable.jl) — same per-bond-sector dimensions,
# same `(Ws, bondsectors)` contract — but keeps an explicit, persistent bipartite graph handed from
# one site step to the next, and suffix-merges the right vertices *incrementally* instead of
# re-materialising every strand's suffix path each bond.
#
# Non-abelian mapping (see research/itensor-mpograph-construction.md §9):
# * Right vertices  = terms of the `ITOTermTable` (persist across the whole sweep; only their count
#   shrinks via suffix-merge). ITensor's "term + additive QN flux" becomes "term + running fusion
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
using SparseArraysBase: SparseArraysBase, SparseMatrixDOK

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
by [`_at_site!`](@ref). Right vertices are terms (identified by a representative term id); they
persist and only shrink via the incremental suffix-merge. The current bipartite graph (for the bond
`i-1 → i` about to be processed) is `lefts` ↔ right vertices, with per-left-vertex adjacency lists
`radj`/`wadj` (right-vertex id, scalar weight).

Fields split into three groups:
* seeding-fixed merge machinery: `sortpos` (right vertices in reversed-suffix sort order) and `lcp`
  (longest-common-prefix of consecutive reversed suffix paths) drive the O(1)-per-boundary
  incremental suffix-merge;
* persistent right-vertex state: `rrepr` (representative term id per right vertex) and `rhi` (the
  max sorted position in each right vertex's merged class, for boundary-lcp lookup);
* the current bipartite graph: `lefts`, `radj`, `wadj`, and `nlinks` (incoming bond dimension).
"""
mutable struct ITOGraph{I <: Sector}
    tt::ITOTermTable{I}
    N::Int
    sortpos::Vector{Int}   # sorted position -> term id (fixed)
    lcp::Vector{Int}       # lcp[k] between sorted positions k and k+1 (length M-1, fixed)
    rrepr::Vector{Int}     # right vertex -> representative term id
    rhi::Vector{Int}       # right vertex -> max sorted position in its merged class
    lefts::Vector{LeftVertex{I}}
    radj::Vector{Vector{Int}}
    wadj::Vector{Vector{ComplexF64}}
    nlinks::Int
end

# Longest common prefix of two equal-length reversed suffix paths (compared as `ITOKey`s).
function _lcp_len(a::Vector{ITOKey{I}}, b::Vector{ITOKey{I}}) where {I}
    n = 0
    @inbounds for k in 1:min(length(a), length(b))
        a[k] == b[k] || break
        n += 1
    end
    return n
end

"""
    ITOGraph(tt::ITOTermTable{I}, N) -> ITOGraph{I}

Seed the persistent graph (mirrors ITensor's `MPOGraph(os)`): materialise each term's full path,
reverse it (descending site) and sort so equal-suffix runs are contiguous — the precondition the
incremental suffix-merge relies on — then bucket the initial left vertices by the site-1 key against
the single left-boundary link. Right vertices start as one-per-term (suffix-from-1 classes are all
distinct, since `ITOTermTable` already merged coincident active content).
"""
function ITOGraph(tt::ITOTermTable{I}, N::Int) where {I}
    M = nterms(tt)
    # full path of each term, reversed to descending site order (site N first)
    rpaths = [ITOKey{I}[_op_at_ito(tt, t, s) for s in N:-1:1] for t in 1:M]
    sortpos = sort(1:M; by = t -> rpaths[t])
    lcp = Int[_lcp_len(rpaths[sortpos[k]], rpaths[sortpos[k + 1]]) for k in 1:(M - 1)]

    rrepr = collect(sortpos)          # right vertex r ↔ sorted position r, repr = sortpos[r]
    rhi = collect(1:M)                # each class is a single sorted position initially

    # seed left vertices: single boundary link, bucketed by the site-1 key
    buckets = Dictionary{ITOKey{I}, Int}()
    lefts = LeftVertex{I}[]
    radj = Vector{Int}[]
    wadj = Vector{ComplexF64}[]
    for r in 1:M
        t = sortpos[r]
        key1 = _op_at_ito(tt, t, 1)
        b = get(buckets, key1, 0)
        if iszero(b)
            push!(lefts, LeftVertex{I}(1, key1))
            push!(radj, Int[])
            push!(wadj, ComplexF64[])
            b = length(lefts)
            insert!(buckets, key1, b)
        end
        push!(radj[b], r)             # right vertex id == sorted position r
        push!(wadj[b], tt.coeffs[t])
    end

    return ITOGraph{I}(tt, N, sortpos, lcp, rrepr, rhi, lefts, radj, wadj, 1)
end

# Phase 1: incrementally suffix-merge right vertices "equal from site i+1 on". Because right vertices
# stay in the seeding sort order and only merge (never split), two currently-adjacent right vertices
# `r, r+1` are equal from i+1 on iff the single boundary `lcp[rhi[r]] >= N - i` (their interiors
# already satisfy the coarser previous threshold). Returns `remap[old_right_id] -> new_right_id` and
# updates `g.rrepr`/`g.rhi` in place.
function _suffix_merge!(g::ITOGraph{I}, i::Int) where {I}
    thr = g.N - i
    R = length(g.rrepr)
    remap = Vector{Int}(undef, R)
    newrrepr = Int[]
    newrhi = Int[]
    r = 1
    while r <= R
        start = r
        while r < R && g.lcp[g.rhi[r]] >= thr
            r += 1
        end
        push!(newrrepr, g.rrepr[start])
        push!(newrhi, g.rhi[r])
        newid = length(newrrepr)
        for rr in start:r
            remap[rr] = newid
        end
        r += 1
    end
    g.rrepr = newrrepr
    g.rhi = newrhi
    return remap
end

# Apply a right-vertex remap to every left vertex's adjacency, summing weights of edges that now land
# on the same merged right vertex.
function _remap_adjacency!(g::ITOGraph, remap::Vector{Int})
    for lv in 1:length(g.lefts)
        acc = Dictionary{Int, ComplexF64}()
        for (rid, w) in zip(g.radj[lv], g.wadj[lv])
            increaseindex!(acc, remap[rid], w)
        end
        g.radj[lv] = collect(keys(acc))
        g.wadj[lv] = collect(values(acc))
    end
    return g
end

# Per-component minimum-vertex-cover backend (the VC path). Given a connected component `(us, vs)`
# (global left/right vertex ids), the scalar coefficient matrix `coeff` and boolean `adjacency`, it
# chooses the component's bond basis via `min_vertex_cover_bipartite` and returns, for that
# component: `rank`, `blocks` (`(incoming_link, local_bond_index, localop)`), `nextedges` (per local
# bond index, the `(right_vertex_id, weight)` edges to forward; empty at the last site) and `secs`
# (the bond charge per local index). Reproduces the covered-U / covered-V coefficient-flow of
# `_irrep_bipartite`.
function _vc_component(
        g::ITOGraph{I}, us::Vector{Int}, vs::Vector{Int},
        coeff::Matrix{ComplexF64}, i::Int
    ) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    N = g.N
    cUbits, cVbits = min_vertex_cover_bipartite((!iszero).(@view coeff[us, vs]))
    cU = findall(cUbits)
    cV = findall(cVbits)
    nleft = length(cU)
    rank = nleft + length(cV)

    blocks = Tuple{Int, Int, LOp}[]
    nextedges = [Tuple{Int, ComplexF64}[] for _ in 1:rank]
    secs = Vector{I}(undef, rank)

    # covered-left vertices → local bond indices 1 … nleft ("a term starts its operator here")
    for (m, lu) in enumerate(cU)
        iu = us[lu]
        lv = g.lefts[iu]
        secs[m] = lv.key.bond
        if i == N
            w = sum(coeff[iu, r] for r in vs)
            push!(blocks, (lv.link, m, lv.key.op * w))
        else
            push!(blocks, (lv.link, m, convert(LOp, lv.key.op)))
            for r in vs
                w = coeff[iu, r]
                iszero(w) || push!(nextedges[m], (r, w))
            end
        end
    end

    # covered-right vertices → local bond indices nleft+1 … rank ("a shared suffix flows through")
    coveredU = falses(length(us))
    coveredU[cU] .= true
    for (p, lvv) in enumerate(cV)
        m = nleft + p
        iv = vs[lvv]
        conn = [us[k] for k in 1:length(us) if !iszero(coeff[us[k], iv])]
        charge = g.lefts[first(conn)].key.bond
        @assert all(g.lefts[iu].key.bond == charge for iu in conn) "bond index not sector-pure (block-diagonality violated)"
        secs[m] = charge
        for k in 1:length(us)
            coveredU[k] && continue
            iu = us[k]
            w = coeff[iu, iv]
            iszero(w) && continue
            push!(blocks, (g.lefts[iu].link, m, g.lefts[iu].key.op * w))
        end
        i == N || push!(nextedges[m], (iv, one(ComplexF64)))
    end

    return rank, blocks, nextedges, secs
end

# Phases 1 & 2, shared by both backends: suffix-merge the right vertices, then materialise the scalar
# coefficient matrix `coeff[u, v]` (summed over parallel edges) and the left-vertex adjacency lists
# `adjU`. Returns `(coeff, adjU, nU, nV)`. (`g.radj` entries are already unique — they are the keys of
# the `Dictionary` built in `_remap_adjacency!` — so `sort` alone suffices for a deterministic order.)
function _bond_matrix!(g::ITOGraph{I}, i::Int) where {I}
    remap = _suffix_merge!(g, i)
    _remap_adjacency!(g, remap)

    nU = length(g.lefts)
    nV = length(g.rrepr)
    coeff = zeros(ComplexF64, nU, nV)
    adjU = Vector{Vector{Int}}(undef, nU)
    for iu in 1:nU
        for (rid, w) in zip(g.radj[iu], g.wadj[iu])
            coeff[iu, rid] += w
        end
        adjU[iu] = sort(g.radj[iu])
    end
    return coeff, adjU, nU, nV
end

# Phase 5, shared by both backends: build the next graph reusing the SAME (persistent) right vertices
# and tagging fresh left vertices with the outgoing bond index `j` as their `link`, bucketed by the
# next-site key `op@(i+1)` of the right vertex. `nextedges_global[j]` is the list of `(right_vertex,
# weight)` edges the outgoing bond index `j` forwards. Mutates `g` into the graph for bond `i → i+1`.
function _build_next_graph!(
        g::ITOGraph{I}, i::Int, nout::Int,
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}}
    ) where {I}
    buckets = Dictionary{Tuple{Int, ITOKey{I}}, Int}()
    next_lefts = LeftVertex{I}[]
    next_radj = Vector{Int}[]
    next_wadj = Vector{ComplexF64}[]
    for j in 1:nout
        for (rid, w) in nextedges_global[j]
            key = _op_at_ito(g.tt, g.rrepr[rid], i + 1)
            b = get(buckets, (j, key), 0)
            if iszero(b)
                push!(next_lefts, LeftVertex{I}(j, key))
                push!(next_radj, Int[])
                push!(next_wadj, ComplexF64[])
                b = length(next_lefts)
                insert!(buckets, (j, key), b)
            end
            push!(next_radj[b], rid)
            push!(next_wadj[b], w)
        end
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

    coeff, adjU, nU, nV = _bond_matrix!(g, i)
    us_of_comp, vs_of_comp = bipartite_connected_components(adjU, nV)

    # phases 3 & 4 — per-component cover, concatenate into the bond
    secW = I[]
    nextedges_global = Vector{Tuple{Int, ComplexF64}}[]
    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    offset = 0
    for (us, vs) in zip(us_of_comp, vs_of_comp)
        rank, blocks, nextedges, secs = _vc_component(g, us, vs, coeff, i)
        for (link, m, op) in blocks
            increaseindex!(site_dict, CartesianIndex(link, offset + m), op)
        end
        for m in 1:rank
            push!(secW, secs[m])
            push!(nextedges_global, nextedges[m])
        end
        offset += rank
    end
    nout = offset

    Ws_i = SparseArraysBase.sparse(site_dict, (g.nlinks, nout))
    i < g.N && _build_next_graph!(g, i, nout, nextedges_global)
    return Ws_i, secW
end

"""
    _irrep_graph_bipartite(tt::ITOTermTable{I}, N) -> (Ws, bondsectors)

Persistent-graph, minimum-vertex-cover reduced-MPO sweep — the ITensor `at_site!` port. Produces the
same `(Ws::Vector{SparseMatrixDOK{LocalOp{ComplexF64, IrrepOperator{I}}}}, bondsectors::Vector{
Vector{I}})` contract as [`_irrep_bipartite`](@ref), so `mpo_terms` / `irrep_mpo_tensors` consume it
unchanged.
"""
function _irrep_graph_bipartite(tt::ITOTermTable{I}, N::Int) where {I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    nterms(tt) == 0 && return (SparseMatrixDOK{LOp}[], Vector{I}[])

    g = ITOGraph(tt, N)
    Ws = Vector{SparseMatrixDOK{LOp}}(undef, N)
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

    coeff, _, nU, nV = _bond_matrix!(g, i)

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

    Ws_i = SparseArraysBase.sparse(site_dict, (g.nlinks, r))
    i < N && _build_next_graph!(g, i, r, nextedges_global)
    return Ws_i, secW
end

"""
    _irrep_graph_svd(tt::ITOTermTable{I}, N, trunc) -> (Ws, bondsectors)

Persistent-graph SVD reduced-MPO sweep — the ITensor QR-backend port. Same `(Ws, bondsectors)`
contract as [`_irrep_graph_bipartite`](@ref) / [`_irrep_svd`](@ref); each bond's compressed basis is
the left singular vectors of the charge-graded bond coefficient matrix, truncated globally across
sectors by `trunc` (a `MatrixAlgebraKit.TruncationStrategy`, or `nothing` for the lossless default).

**Lossless (`trunc === nothing`) this is at parity with [`_irrep_svd`](@ref)** — same per-sector bond
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
    nterms(tt) == 0 && return (SparseMatrixDOK{LOp}[], Vector{I}[])

    truncstrat = something(trunc, trunctol(rtol = eps(Float64)))
    g = ITOGraph(tt, N)
    Ws = Vector{SparseMatrixDOK{LOp}}(undef, N)
    bondsectors = Vector{Vector{I}}(undef, N)
    for i in 1:N
        Ws[i], bondsectors[i] = _svd_at_site!(g, i, truncstrat)
    end
    return (Ws, bondsectors)
end
