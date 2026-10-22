# VertexCover bond-basis backend: per-component minimum vertex cover, plus the finish-channel
# forcing Jordan emission needs.

# Move the finish class (local index `pf`) into the cover and drop left vertices that makes redundant.
# A minimum cover may cover it from the left, whose index would emit the last factor's letter rather
# than the identity; Jordan form needs the covered-right reading. Never costs: the cover grows by at
# most one and saves a padded index. The start channel needs no counterpart: König never covers the
# degree-one sentinel, so its left vertex is covered-left, and it is not adjacent to the finish class.
function _force_finish!(cUbits, cVbits, localadj::Vector{Vector{Int}}, pf::Int)
    cVbits[pf] && return false
    cVbits[pf] = true
    @inbounds for k in eachindex(localadj)
        cUbits[k] || continue
        all(q -> cVbits[q], localadj[k]) && (cUbits[k] = false)
    end
    return true
end

# Minimum-vertex-cover backend for component `(us, vs)` (global ids). Returns `rank`, `blocks`
# `(incoming_link, local_bond_index, op)`, `nextedges` (per local index, the `(right vertex, weight)`
# edges to forward; empty at the last site), `secs`, `startidx`/`finishidx` (0 if not here) and
# `origins` (`(0, left)` / `(1, right)` per index, for canonical ordering). Covered-left forwards its
# weights and emits the bare letter; covered-right resets the weight to 1 and folds `op × weight` into
# every uncovered neighbour's block. `Θ(E_component)`.
#
# Finish channel: only `g.rfinish` or `g.finishleft` can carry it; with `g.jordan` `_force_finish!`
# covers `rfinish` so the pass-through folds in, otherwise whichever the cover takes is read. The
# sentinel is never covered (see `_force_finish!`); the covered-right case is its unreachable dual.
function _vc_component(
        g::SweepGraph{I}, us::Vector{Int}, vs::Vector{Int}, i::Int
    ) where {I}
    LOp = SiteOperator{I}
    N = g.N
    nus, nvs = length(us), length(vs)

    # component-local right-vertex numbering
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
    origins = Vector{Tuple{Int, Int}}(undef, rank)
    startidx = 0
    finishidx = 0

    # covered-left vertices → indices 1 … nleft
    for (m, lu) in enumerate(cU)
        iu = us[lu]
        lv = g.lefts[iu]
        secs[m] = lv.key.bond
        origins[m] = (0, iu)
        iu == g.startleft && (startidx = m)
        iu == g.finishleft && (finishidx = m)
        if i == N
            iszero(g.rsent) || invariant_violation("the sentinel must be gone by the last site")
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

    # covered-right vertices → indices nleft+1 … rank; `bondof[p]` is 0 if not covered
    bondof = zeros(Int, nvs)
    for (p, lvv) in enumerate(cV)
        m = nleft + p
        iv = vs[lvv]
        bondof[lvv] = m
        origins[m] = (1, iv)
        # components are charge-pure (`_prepare_bond!`), so any incident left gives the charge
        secs[m] = g.lefts[g.firstleft[iv]].key.bond
        if iv == g.rfinish
            iszero(finishidx) ||
                invariant_violation("finish channel covered on both sides (the cover is not minimum)")
            finishidx = m
        end
        if iv == g.rsent
            iszero(startidx) || invariant_violation("start channel covered on both sides")
            startidx = m                        # the sentinel *is* the start channel here
        elseif i != N
            push!(nextedges[m], (iv, one(ComplexF64)))
        end
    end

    # every neighbour of an uncovered left vertex is covered: fold the coefficients
    coveredU = falses(nus)
    coveredU[cU] .= true
    for k in 1:nus
        coveredU[k] && continue
        iu = us[k]
        lv = g.lefts[iu]
        for (rid, w) in zip(g.radj[iu], g.wadj[iu])
            iszero(w) && continue
            m = bondof[vlocal[rid]]
            iszero(m) && invariant_violation("edge left uncovered by the minimum vertex cover")
            push!(blocks, (lv.link, m, lv.key.op * w))
        end
    end

    return rank, blocks, nextedges, secs, startidx, finishidx, origins
end

# Minimum vertex cover of one component with every cyclic channel state forced into it: otherwise the
# predecessor is covered-left and forwards `λ·w` instead of landing `λ · pass-through` on the diagonal.
# Free in the bulk (the state has a pendant predecessor); at most one index in a window's boundary cells.
function _forced_cover(
        g::SweepGraph, localadj::Vector{Vector{Int}}, nus::Int, nvs::Int, vs::Vector{Int}
    )
    forced = BitVector(_iscyclic(g, vs[p]) for p in 1:nvs)
    any(forced) || return minimum_vertex_cover(localadj, nus, nvs)
    residual = [filter(p -> !forced[p], adj) for adj in localadj]
    cUbits, cVbits = minimum_vertex_cover(residual, nus, nvs)
    return cUbits, cVbits .| forced
end

# Reorder the assembled bond canonically: covered-left by `(link, key)`, then covered-right by `_rdesc`.
# Both are translation-invariant given a canonical incoming bond, so bonds `i` and `i+L` of a periodic
# model get identical bases, not merely isomorphic ones.
function _canonicalise_bond!(
        g::SweepGraph{I}, i::Int, nout::Int, site_dict, secW::Vector{I},
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

# `_bond_basis!` for `VertexCover`: cover each connected component and concatenate the ranks.
# Returns `(nout, site_dict, secW, nextedges_global)`.
function _bond_basis!(g::SweepGraph{I}, i::Int, nU::Int, nV::Int, ::VertexCover) where {I}
    LOp = SiteOperator{I}
    us_of_comp, vs_of_comp = connected_components(g.radj, nV)

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
                invariant_violation("the start channel appeared in more than one component")
            g.startidx = offset + startidx
        end
        if !iszero(finishidx)
            iszero(g.finishidx) ||
                invariant_violation("the finish channel appeared in more than one component")
            g.finishidx = offset + finishidx
        end
        offset += rank
    end
    nout = offset
    site_dict, secW, nextedges_global = _canonicalise_bond!(
        g, i, nout, site_dict, secW, nextedges_global, origins_global
    )
    (iszero(g.startidx) || g.startidx != g.finishidx) ||
        invariant_violation("the start and finish channels resolved to the same bond index")
    return nout, site_dict, secW, nextedges_global
end
