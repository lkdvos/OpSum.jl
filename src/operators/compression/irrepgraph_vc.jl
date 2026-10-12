# VertexCover bond-basis backend: per-component minimum vertex cover, plus the finish-channel
# forcing Jordan emission needs.

"""
    _force_finish!(cUbits, cVbits, localadj, pf) -> Bool

Move the finish class (component-local right index `pf`) into the cover, and drop every left vertex
that this makes redundant. Returns whether anything changed.

A minimum cover may cover the finish class from the *left* instead (when it and one left vertex form
an isolated matched pair, König visits neither), and that index would emit the last factor's letter,
not the identity. Jordan form needs a channel emitting exactly `1 · id` from the previous finish
index, i.e. the covered-*right* reading. Forcing never costs: the cover grows by at most one, and it
saves the padded index it would otherwise need. Dropping a left vertex whose neighbours are all
covered is safe.

The start channel needs no counterpart: König never covers a degree-one right vertex, so the
sentinel's left vertex `L₀` is always covered-left (see §2.2 of research/persistent-graph-mpo.md),
and it is never adjacent to the finish class, so forcing cannot remove it.
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

# Per-component minimum-vertex-cover backend for the component `(us, vs)` (global left/right ids).
# Returns `rank`, `blocks` (`(incoming_link, local_bond_index, localop)`), `nextedges` (per local bond
# index, the `(right_vertex_id, weight)` edges to forward; empty at the last site), `secs` (bond
# charge per index), `startidx`/`finishidx` (local index of the identity/start and finish channel, 0
# if not in this component) and `origins` (the vertex each index came from, for canonical ordering).
# Covered-left forwards its edge weights unchanged and emits the bare letter; covered-right resets
# the forwarded weight to 1 and folds `key.op × weight` into the block of every uncovered incident
# left. Cost is `Θ(E_component)` off `g.radj`/`g.wadj`/`g.firstleft`.
#
# Finish channel: Jordan form needs an index meaning "every factor is placed", reachable only from the
# previous finish index. Exactly two vertices can carry it: the right vertex `g.rfinish` and the left
# vertex `g.finishleft` (previous finish index, pass-through), whose only neighbour is `g.rfinish`.
# With `g.jordan`, `_force_finish!` covers `g.rfinish`, so `g.finishleft` is uncovered and folds
# `passthrough × 1` in; otherwise whichever the cover takes is read, never both. Either reading emits
# the bare pass-through with weight 1, so the `(end, end)` corner is an exact identity.
#
# Sentinel: a degree-1 right vertex, which König never covers (`L₀` is matched, and the sentinel is
# reachable only from `L₀`), so `L₀` is always covered-left and emits the bare pass-through. The
# covered-right sentinel case below is unreachable; it is the exact dual (an uncovered `L₀` folds
# `passthrough × 1` into the same block), kept so a change to the cover cannot silently lose the
# identity channel. Cyclic channel states are forced into the cover, see `_forced_cover`.
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
    # `origins[m]`: the vertex index `m` came from, `(0, left)` or `(1, right)`
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
        # the component is charge-pure (checked in `_prepare_bond!`), so any incident left gives it
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

    # every neighbour of an uncovered left vertex is a covered right vertex: fold their coefficients
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
# A cyclic state is live at every bond and re-enters itself. Left uncovered, its predecessor becomes
# covered-left and *forwards* the self-edge weight `λ·w` instead of resetting it to 1, so λ powers ride
# along the bond rather than landing on the channel's diagonal. König can pick that cover (two minimum
# covers tie once the channel's entry letter is shared with a finite-range term); the sweep heals one
# bond later either way, but forcing makes convergence structural and two bonds earlier.
#
# It is free: `{v} ∪ MVC(G ∖ v)` is minimum among covers containing `v`, and in the bulk a cyclic state
# has a pendant predecessor, so exchanging that for the state is never worse. It can cost one extra
# index only in a window's discarded boundary cells. Removing the forced columns also makes the
# predecessor come back *uncovered*, which folds `λ · pass-through` onto the diagonal.
function _forced_cover(
        g::ITOGraph, localadj::Vector{Vector{Int}}, nus::Int, nvs::Int, vs::Vector{Int}
    )
    forced = BitVector(_iscyclic(g, vs[p]) for p in 1:nvs)
    any(forced) || return min_vertex_cover_bipartite(localadj, nus, nvs)[1:2]
    residual = [filter(p -> !forced[p], adj) for adj in localadj]
    cUbits, cVbits = min_vertex_cover_bipartite(residual, nus, nvs)
    return cUbits, cVbits .| forced
end

"""
    _canonicalise_bond!(g, i, nout, site_dict, secW, nextedges_global, origins_global)

Reorder the assembled bond into canonical order: covered-left indices first, ordered by
`(incoming bond index, on-site ITOKey)`, then covered-right indices by their class name
[`_rdesc`](@ref). Permutes the block dictionary's columns, the bond charges, the forwarded edges and
`g.startidx`/`g.finishidx` together.

Both keys are translation-invariant given that the incoming bond was canonically ordered (induction
from the 1-dimensional boundary bond), so bond `i` and bond `i+L` of a periodic model get identically
labelled bases, not merely isomorphic ones (see [`_canonicalise_rights!`](@ref)).
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

# Phases 3 & 4 for [`VertexCover`](@ref): split the bond into connected components, cover each
# (`_vc_component`) and concatenate the ranks into one bond. Returns the `_bond_basis!` contract
# `(nout, site_dict, secW, nextedges_global)`.
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
