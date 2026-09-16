# The SVD bond-basis backends: SequentialSVD (rides the persistent graph, dense per-bond SVD)
# and IndependentSVD (compresses each bond independently, sharing only the interning helpers).

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

# Shared by `_bond_basis!(..., SequentialSVD)` and `_irrep_independent_svd`'s phase 1: SVD a
# charge-graded coefficient block `C : Ppre ← Psuf` (with `trunc`) and flatten the retained left
# singular vectors `U` into a dense `(nrow × r)` matrix grouped by sector, recording the per-column
# charge `secs`; `Umat[p, col] = block(U, secs[col])[rowdeg[p], localcol]` for every `p` with
# `rowQ[p] == secs[col]`. Passing `colQ`/`coldeg` (and, where not every column is populated, `colactive`)
# additionally flattens `R = S · Vᴴ` into `Rmat`, the compressed suffix side `SequentialSVD` forwards
# onto the next bond; `_irrep_independent_svd` has no next bond to forward to and omits them.
function _svd_flatten_block(
        C, trunc, rowQ::Vector{I}, rowdeg::Vector{Int};
        colQ::Union{Nothing, Vector{I}} = nothing, coldeg::Union{Nothing, Vector{Int}} = nothing,
        colactive::Union{Nothing, BitVector} = nothing
    ) where {I}
    U, S, Vt = svd_trunc(C; trunc)
    Wb = space(S, 1)   # retained bond space (⊕ charge sectors, truncated multiplicities)
    T = eltype(U)
    nrow = length(rowQ)
    r = sum(q -> dim(Wb, q), sectors(Wb); init = 0)
    Umat = zeros(T, nrow, r)
    secs = Vector{I}(undef, r)
    R = colQ === nothing ? nothing : S * Vt
    Rmat = colQ === nothing ? nothing : zeros(T, r, length(colQ))
    col = 0
    for q in sectors(Wb)
        Ub = block(U, q)
        Rb = colQ === nothing ? nothing : block(R, q)
        for localcol in 1:size(Ub, 2)
            col += 1
            secs[col] = q
            for p in 1:nrow
                rowQ[p] == q || continue
                Umat[p, col] = Ub[rowdeg[p], localcol]
            end
            if colQ !== nothing
                for v in eachindex(colQ)
                    active = colactive === nothing || colactive[v]
                    (active && colQ[v] == q) || continue
                    Rmat[col, v] = Rb[localcol, coldeg[v]]
                end
            end
        end
    end
    col == r || _invariant("retained bond space does not match its per-sector dimensions")
    return (; U = Umat, secs, R = Rmat)
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

    flat = _svd_flatten_block(
        C, strategy.trunc, uCharge, udeg;
        colQ = vCharge, coldeg = vdeg, colactive = vactive
    )
    Umat, secW, Rmat = flat.U, flat.secs, flat.R   # column m = compressed prefix basis vector
    r = length(secW)

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

_resolve_trunc(s::SequentialSVD) = SequentialSVD(something(s.trunc, trunctol(rtol = eps(Float64))))

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

        flat = _svd_flatten_block(C, truncstrat, preQ, pre_deg)
        bond_Us[b] = flat.U
        bond_secs[b] = flat.secs
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

function _irrep_sweep(tt::ITOTermTable, N::Int, strategy::IndependentSVD)
    return _irrep_independent_svd(tt, N, strategy.trunc)
end

# An SVD bond basis is a mixture of prefix states; a geometric channel is a bond index with a diagonal
# rather than a column of one, so there is nothing to mix it into.
function _irrep_sweep(
        ::ITOTermTable{I}, ::Int, ::IndependentSVD; channels::Vector{ExpChannel{I}}
    ) where {I}
    return isempty(channels) || throw(
        ArgumentError(
            "IndependentSVD cannot compress an exponentially decaying interaction: it needs the " *
                "whole bond coefficient matrix up front. Use BipartiteAlgorithm() (the default)."
        )
    )
end

function _irrep_channels(tt::ITOTermTable, N::Int, strategy::IndependentSVD)
    Ws, bondsectors = _irrep_independent_svd(tt, N, strategy.trunc)
    return (Ws, bondsectors, zeros(Int, length(Ws)), zeros(Int, length(Ws)))
end
