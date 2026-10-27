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

# The cover of one component: `rank` indices, `blocks` `(incoming_link, local_bond_index, op)`, `nextedges`
# (per local index, the `(right vertex, weight)` edges to forward; empty at the last site), `secs`,
# `startidx`/`finishidx` (0 if not here) and `origins` (`(0, left)` / `(1, right)` per index, for canonical
# ordering).
struct ComponentCover{I <: Sector}
    rank::Int
    blocks::Vector{Tuple{Int, Int, SiteOperator{I}}}
    nextedges::Vector{Vector{Tuple{Int, ComplexF64}}}
    secs::Vector{I}
    startidx::Int
    finishidx::Int
    origins::Vector{Tuple{Int, Int}}
end

# Minimum-vertex-cover backend for component `(us, vs)` (global ids). Covered-left forwards its
# weights and emits the bare letter; covered-right resets the weight to 1 and folds `op × weight` into
# every uncovered neighbour's block. `Θ(E_component)`.
#
# Finish channel: only `g.bound.rfinish` or `g.bound.finishleft` can carry it; with `g.jordan` `_force_finish!`
# covers `rfinish` so the pass-through folds in, otherwise whichever the cover takes is read. The
# sentinel is never covered (see `_force_finish!`); the covered-right case is its unreachable dual.
function _vc_component(
        g::SweepGraph{I}, us::Vector{Int}, vs::Vector{Int}, i::Int
    ) where {I}
    LOp = SiteOperator{I}
    N = g.N
    nus, nvs = length(us), length(vs)

    # component-local right-vertex numbering
    vlocal = g.scratch.vlocal
    @inbounds for p in 1:nvs
        vlocal[vs[p]] = p
    end
    localadj = Vector{Vector{Int}}(undef, nus)
    @inbounds for k in 1:nus
        radj = g.left.radj[us[k]]
        localadj[k] = Int[vlocal[rid] for rid in radj]
    end

    cUbits, cVbits = _forced_cover(g, localadj, nus, nvs, vs)
    if g.jordan && !iszero(g.bound.rfinish)
        pf = vlocal[g.bound.rfinish]                # stale unless `rfinish` is in this component
        (1 <= pf <= nvs && vs[pf] == g.bound.rfinish) && _force_finish!(cUbits, cVbits, localadj, pf)
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
        lv = g.left.vertex[iu]
        secs[m] = lv.key.bond
        origins[m] = (0, iu)
        iu == g.bound.startleft && (startidx = m)
        iu == g.bound.finishleft && (finishidx = m)
        if i == N
            iszero(g.bound.rsent) || invariant_violation("the sentinel must be gone by the last site")
            push!(blocks, (lv.link, m, lv.key.letter * sum(g.left.wadj[iu]; init = zero(ComplexF64))))
        else
            push!(blocks, (lv.link, m, convert(LOp, lv.key.letter)))
            edges = nextedges[m]
            for (rid, w) in zip(g.left.radj[iu], g.left.wadj[iu])
                (rid == g.bound.rsent || iszero(w)) && continue   # the sentinel is regenerated, not carried
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
        secs[m] = g.left.vertex[g.scratch.firstleft[iv]].key.bond
        if iv == g.bound.rfinish
            iszero(finishidx) ||
                invariant_violation("finish channel covered on both sides (the cover is not minimum)")
            finishidx = m
        end
        if iv == g.bound.rsent
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
        lv = g.left.vertex[iu]
        for (rid, w) in zip(g.left.radj[iu], g.left.wadj[iu])
            iszero(w) && continue
            m = bondof[vlocal[rid]]
            iszero(m) && invariant_violation("edge left uncovered by the minimum vertex cover")
            push!(blocks, (lv.link, m, lv.key.letter * w))
        end
    end

    return ComponentCover{I}(rank, blocks, nextedges, secs, startidx, finishidx, origins)
end

# Minimum vertex cover of one component with every cyclic channel state forced into it: otherwise the
# predecessor is covered-left and forwards `λ·w` instead of landing `λ · pass-through` on the diagonal.
# Free in the bulk (the state has a pendant predecessor); at most one index in a window's boundary cells.
function _forced_cover(
        g::SweepGraph, localadj::Vector{Vector{Int}}, nus::Int, nvs::Int, vs::Vector{Int}
    )
    forced = BitVector(iscyclic(g, vs[p]) for p in 1:nvs)
    any(forced) || return minimum_vertex_cover(localadj, nus, nvs)
    residual = [filter(p -> !forced[p], adj) for adj in localadj]
    cUbits, cVbits = minimum_vertex_cover(residual, nus, nvs)
    return cUbits, cVbits .| forced
end

# Reorder the assembled bond canonically: covered-left by `(link, key)`, then covered-right by `rdesc`.
# Both are translation-invariant given a canonical incoming bond, so bonds `i` and `i+L` of a periodic
# model get identical bases, not merely isomorphic ones.
function _canonicalise_bond(
        g::SweepGraph{I}, i::Int, b::BondBasis{I}, origins::Vector{Tuple{Int, Int}}
    ) where {I}
    nout = b.nout
    nout <= 1 && return b
    covleft = [m for m in 1:nout if iszero(origins[m][1])]
    covright = [m for m in 1:nout if isone(origins[m][1])]
    sort!(covleft; by = m -> (lv = g.left.vertex[origins[m][2]]; (lv.link, lv.key)))
    sort!(covright; by = m -> rdesc(g, origins[m][2], i))
    order = vcat(covleft, covright)
    issorted(order) && return b

    pos = Vector{Int}(undef, nout)
    for (k, m) in enumerate(order)
        pos[m] = k
    end
    newdict = Dictionary{CartesianIndex{2}, SiteOperator{I}}()
    for (idx, op) in pairs(b.site)
        l, m = Tuple(idx)
        increaseindex!(newdict, CartesianIndex(l, pos[m]), op)
    end
    startidx = iszero(b.startidx) ? 0 : pos[b.startidx]
    finishidx = iszero(b.finishidx) ? 0 : pos[b.finishidx]
    return BondBasis{I}(nout, newdict, b.sectors[order], b.nextedges[order], startidx, finishidx)
end

# `bond_basis!` for `VertexCover`: cover each connected component and concatenate the ranks.
function bond_basis!(g::SweepGraph{I}, i::Int, nU::Int, nV::Int, ::VertexCover) where {I}
    LOp = SiteOperator{I}
    us_of_comp, vs_of_comp = connected_components(g.left.radj, nV)

    secW = I[]
    nextedges_global = Vector{Tuple{Int, ComplexF64}}[]
    origins_global = Tuple{Int, Int}[]
    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    startidx = finishidx = 0
    offset = 0
    for (us, vs) in zip(us_of_comp, vs_of_comp)
        comp = _vc_component(g, us, vs, i)
        for (link, m, op) in comp.blocks
            increaseindex!(site_dict, CartesianIndex(link, offset + m), op)
        end
        for m in 1:comp.rank
            push!(secW, comp.secs[m])
            push!(nextedges_global, comp.nextedges[m])
            push!(origins_global, comp.origins[m])
        end
        if !iszero(comp.startidx)
            iszero(startidx) ||
                invariant_violation("the start channel appeared in more than one component")
            startidx = offset + comp.startidx
        end
        if !iszero(comp.finishidx)
            iszero(finishidx) ||
                invariant_violation("the finish channel appeared in more than one component")
            finishidx = offset + comp.finishidx
        end
        offset += comp.rank
    end
    b = _canonicalise_bond(
        g, i, BondBasis{I}(offset, site_dict, secW, nextedges_global, startidx, finishidx), origins_global
    )
    (iszero(b.startidx) || b.startidx != b.finishidx) ||
        invariant_violation("the start and finish channels resolved to the same bond index")
    return b
end
