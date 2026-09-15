# VertexCover bond-basis backend: per-component minimum vertex cover, plus the finish-channel
# forcing Jordan emission needs.

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
#
# The one class of right vertex that is *forced* into the cover is a cyclic channel state
# (`_iscyclic`), for the reason spelled out in `_forced_cover`.
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

    cUbits, cVbits = _forced_cover(g, localadj, nus, nvs, vs)
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
    # `origins[m]` names the vertex bond index `m` came from: `(0, left vertex)` or `(1, right vertex)`.
    # `_at_site!` uses it to put the assembled bond into canonical order.
    origins = Vector{Tuple{Int, Int}}(undef, rank)
    startidx = 0
    finishidx = 0

    # covered-left vertices → local bond indices 1 … nleft ("a term starts its operator here")
    for (m, lu) in enumerate(cU)
        iu = us[lu]
        lv = g.lefts[iu]
        secs[m] = lv.key.bond
        origins[m] = (0, iu)
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
        origins[m] = (1, iv)
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

    return rank, blocks, nextedges, secs, startidx, finishidx, origins
end

# Minimum vertex cover of one component, with every *cyclic* channel state forced into it.
#
# Why force. A cyclic state is live at every bond and re-enters itself. If it is left uncovered, its
# predecessor becomes covered-left instead, which *forwards* the self-edge weight `λ·w` rather than
# resetting it to 1 — so the λ powers ride along the bond instead of landing on the channel's diagonal,
# and a bond that keeps doing that never repeats itself. König genuinely can pick that cover: two
# equal-size minimum covers exist as soon as the channel's entry letter is shared with a finite-range
# term (an exp tail alongside the nearest-neighbour term of the same operator), and the matching decides
# which. Measured on that model the choice is made at the *first* bond where both classes are live, and
# it heals one bond later — a covered-left predecessor is itself a second predecessor of the cyclic
# state, which then has two pendants and must be covered. So the sweep converges either way on every
# model here; what forcing buys is that it does so *structurally*, without resting on that
# self-healing argument, and two bonds earlier.
#
# Why it is free. `{v} ∪ MVC(G ∖ v)` is a minimum cover *among the covers containing `v`*, and in the
# bulk a cyclic state always has a pendant predecessor (the bond index it came from forwards nothing
# else), so exchanging that predecessor for the state is never worse — the forced cover is a minimum
# cover outright. It can cost one extra index only where the state has just been created and its only
# predecessor is shared, i.e. in the window's discarded boundary cells; no bond dimension in the test
# suite changes either way. Since a minimum cover keeps no redundant left vertex, removing the forced
# columns also guarantees the predecessor comes back *uncovered*, which is exactly what folds
# `λ · pass-through` onto the diagonal.
function _forced_cover(
        g::ITOGraph, localadj::Vector{Vector{Int}}, nus::Int, nvs::Int, vs::Vector{Int}
    )
    forced = falses(nvs)
    nforced = 0
    @inbounds for p in 1:nvs
        if _iscyclic(g, vs[p])
            forced[p] = true
            nforced += 1
        end
    end
    if iszero(nforced)
        cUbits, cVbits = min_vertex_cover_bipartite(localadj, nus, nvs)
        return cUbits, cVbits
    end
    residual = Vector{Vector{Int}}(undef, nus)
    @inbounds for k in 1:nus
        residual[k] = filter(p -> !forced[p], localadj[k])
    end
    cUbits, cVbits = min_vertex_cover_bipartite(residual, nus, nvs)
    return cUbits, cVbits .| forced
end

"""
    _canonicalise_bond!(g, i, nout, site_dict, secW, nextedges_global, origins_global)

Reorder the assembled bond into canonical order: covered-left indices first, ordered by
`(incoming bond index, on-site ITOKey)`, then covered-right indices ordered by their canonical class
name [`_rdesc`](@ref). Permutes the block dictionary's columns, the bond charges, the forwarded edges
and `g.startidx`/`g.finishidx` together.

Both sort keys are translation-invariant *given* that the incoming bond was itself canonically
ordered — a left vertex is uniquely named by `(link, key)`, and after this pass an index's position
*is* its canonical rank, so `link` needs no further translation. That induction, seeded by the
1-dimensional boundary bond, is what makes bond `i` and bond `i+L` of a periodic model produce
identically labelled bases rather than merely isomorphic ones (see [`_canonicalise_rights!`](@ref) for
why that matters and what the other half of it is).
"""
function _canonicalise_bond!(
        g::ITOGraph{I}, i::Int, nout::Int, site_dict, secW::Vector{I},
        nextedges_global::Vector{Vector{Tuple{Int, ComplexF64}}},
        origins_global::Vector{Tuple{Int, Int}}
    ) where {I}
    nout <= 1 && return site_dict, secW, nextedges_global
    covleft = [m for m in 1:nout if iszero(origins_global[m][1])]
    covright = [m for m in 1:nout if isone(origins_global[m][1])]
    sort!(covleft; by = m -> (lv = g.lefts[origins_global[m][2]]; (lv.link, lv.key)))
    sort!(covright; by = m -> _rdesc(g, origins_global[m][2], i))
    order = vcat(covleft, covright)
    issorted(order) && return site_dict, secW, nextedges_global

    pos = Vector{Int}(undef, nout)
    for (k, m) in enumerate(order)
        pos[m] = k
    end
    newdict = Dictionary{CartesianIndex{2}, valtype(site_dict)}()
    for (idx, op) in pairs(site_dict)
        l, m = Tuple(idx)
        increaseindex!(newdict, CartesianIndex(l, pos[m]), op)
    end
    iszero(g.startidx) || (g.startidx = pos[g.startidx])
    iszero(g.finishidx) || (g.finishidx = pos[g.finishidx])
    return newdict, secW[order], nextedges_global[order]
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
    origins_global = Tuple{Int, Int}[]
    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    offset = 0
    for (us, vs) in zip(us_of_comp, vs_of_comp)
        rank, blocks, nextedges, secs, startidx, finishidx, origins =
            _vc_component(g, us, vs, i)
        for (link, m, op) in blocks
            increaseindex!(site_dict, CartesianIndex(link, offset + m), op)
        end
        for m in 1:rank
            push!(secW, secs[m])
            push!(nextedges_global, nextedges[m])
            push!(origins_global, origins[m])
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
    nout = offset
    site_dict, secW, nextedges_global = _canonicalise_bond!(
        g, i, nout, site_dict, secW, nextedges_global, origins_global
    )
    (iszero(g.startidx) || g.startidx != g.finishidx) ||
        _invariant("the start and finish channels resolved to the same bond index")
    return nout, site_dict, secW, nextedges_global
end
