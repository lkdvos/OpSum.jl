# The SVD bond-basis backends: SequentialSVD (rides the persistent graph, dense per-bond SVD)
# and IndependentSVD (compresses each bond independently, sharing only the interning helpers).

_nextdeg!(mult, q) = mult[q] = get(mult, q, 0) + 1

# Dense class of `key` in `dense`, created on a miss with the next degeneracy index of charge `q`.
function _class!(dense, Qs, deg, mult, key, q)
    c = get(dense, key, 0)
    if iszero(c)
        push!(Qs, q)
        push!(deg, _nextdeg!(mult, q))
        c = length(Qs)
        insert!(dense, key, c)
    elseif Qs[c] != q
        invariant_violation("class not sector-pure")
    end
    return c
end

# `NoTruncation` is lossless up to machine precision: it truncates at `eps(Float64)`, relative to the norm of
# the singular values.
_svd_trunc(trunc::TruncationStrategy) = trunc
_svd_trunc(::NoTruncation) = trunctol(rtol = eps(Float64))

# The retained `U : Ppre ← Wb` of a block SVD as a dense `Umat[p, col]` with column charges `secs`; `p`
# runs over the rows with charge `rowQ[p]` and degeneracy index `rowdeg[p]`.
function _flatten_U(U, Wb, rowQ::Vector{I}, rowdeg::Vector{Int}) where {I}
    T = eltype(U)
    r = sum(q -> dim(Wb, q), sectors(Wb); init = 0)
    Umat = zeros(T, length(rowQ), r)
    secs = Vector{I}(undef, r)
    col = 0
    for q in sectors(Wb)
        Ub = block(U, q)
        for localcol in 1:size(Ub, 2)
            col += 1
            secs[col] = q
            for p in eachindex(rowQ)
                rowQ[p] == q || continue
                Umat[p, col] = Ub[rowdeg[p], localcol]
            end
        end
    end
    col == r || invariant_violation("retained bond space does not match its per-sector dimensions")
    return Umat, secs
end

# `R = S · Vᴴ : Wb ← Psuf` as a dense `Rmat[col, v]`, columns in the order of `_flatten_U`; `v` runs over
# the active columns with charge `colQ[v]` and degeneracy index `coldeg[v]`.
function _flatten_R(R, Wb, colQ::Vector{I}, coldeg::Vector{Int}, colactive::BitVector) where {I}
    r = sum(q -> dim(Wb, q), sectors(Wb); init = 0)
    Rmat = zeros(eltype(R), r, length(colQ))
    col = 0
    for q in sectors(Wb)
        Rb = block(R, q)
        for localcol in 1:size(Rb, 1)
            col += 1
            for v in eachindex(colQ)
                (colactive[v] && colQ[v] == q) || continue
                Rmat[col, v] = Rb[localcol, coldeg[v]]
            end
        end
    end
    return Rmat
end

# `bond_basis!` for `SequentialSVD`: the whole bond's coefficient matrix is a charge-graded
# `C : Ppre ← Psuf`; its truncated `U` is the compressed basis (index `m` emits `key.letter * U[u, m]`) and
# `R = S·Vᴴ` becomes the forwarded edge weights (folded into the block at the last site).
function bond_basis!(g::SweepGraph{I}, i::Int, nU::Int, nV::Int, strategy::SequentialSVD) where {I}
    LOp = SiteOperator{I}
    N = g.N

    # bond charge of each prefix and suffix state (a right vertex orphaned by truncation has none)
    uCharge = I[g.left.vertex[u].key.bond for u in 1:nU]
    vCharge = Vector{I}(undef, nV)
    vactive = falses(nV)
    for u in 1:nU
        for (rid, w) in zip(g.left.radj[u], g.left.wadj[u])
            iszero(w) && continue
            if vactive[rid]
                vCharge[rid] == uCharge[u] ||
                    invariant_violation("bond index not sector-pure (block-diagonality violated)")
            else
                vCharge[rid] = uCharge[u]
                vactive[rid] = true
            end
        end
    end

    # degeneracy index within the charge sector, and per-sector multiplicities
    umult, vmult = Dict{I, Int}(), Dict{I, Int}()
    udeg = [_nextdeg!(umult, q) for q in uCharge]
    vdeg = [vactive[v] ? _nextdeg!(vmult, vCharge[v]) : 0 for v in 1:nV]

    C = zeros(ComplexF64, Vect[I](umult) ← Vect[I](vmult))
    for u in 1:nU
        Cb = block(C, uCharge[u])
        for (rid, w) in zip(g.left.radj[u], g.left.wadj[u])
            iszero(w) || (Cb[udeg[u], vdeg[rid]] += w)
        end
    end

    U, S, Vt = svd_trunc(C; trunc = _svd_trunc(strategy.trunc))
    Wb = space(S, 1)   # retained bond space
    Umat, secW = _flatten_U(U, Wb, uCharge, udeg)   # column m = compressed prefix basis vector
    Rmat = _flatten_R(S * Vt, Wb, vCharge, vdeg, vactive)
    r = length(secW)

    site_dict = Dictionary{CartesianIndex{2}, LOp}()
    nextedges_global = [Tuple{Int, ComplexF64}[] for _ in 1:r]
    for m in 1:r
        for u in 1:nU
            w = Umat[u, m]
            iszero(w) && continue
            lv = g.left.vertex[u]
            # last site: the single trivial suffix folds R into the block
            increaseindex!(site_dict, CartesianIndex(lv.link, m), lv.key.letter * (i == N ? w * Rmat[m, 1] : w))
        end
        if i < N
            for v in 1:nV
                rw = Rmat[m, v]
                iszero(rw) || push!(nextedges_global[m], (v, rw))
            end
        end
    end

    return BondBasis{I}(r, site_dict, secW, nextedges_global)
end

# `IndependentSVD`: not a graph sweep, since each bond is compressed on the raw prefix/suffix classes
# (`prefix_ids`; `(suffix_ids, charge)`) regardless of its neighbours' bases (so it matches `SequentialSVD`
# in bond dimension when lossless, not in gauge). One charge-graded
# `C_b : Ppre ← Psuf` per bond, `U_b` as its basis, then `W_op = U_{i-1}' · C_op · U_i` per letter.
function sweep(tt::TermTable{I}, N::Int, strategy::IndependentSVD) where {I}
    trunc = strategy.trunc
    T = ComplexF64
    Op = Factor{I}
    LOp = SiteOperator{I}
    M = nterms(tt)
    M == 0 && return emptysweep(I)

    K = arity(tt)
    preid = prefix_ids(tt)
    sufid = suffix_ids(tt)
    nb = max(N - 1, 0)

    # 1. per internal bond: `bond_Us[b]` the (n_pre × r_b) isometry, `bond_secs[b]` its column charges,
    #    `predense[b]` interned prefix id -> dense column
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
            q = iszero(j) ? unit(I) : tt.factors[j, t].bond
            pterm[t] = _class!(pd, preQ, pre_deg, pre_mult, preid[j + 1, t], q)
            sterm[t] = _class!(sufdense, sufQ, suf_deg, suf_mult, (sufid[j + 1, t], q), q)
        end

        C = zeros(T, Vect[I](pre_mult) ← Vect[I](suf_mult))
        for t in 1:M
            p, s = pterm[t], sterm[t]
            block(C, preQ[p])[pre_deg[p], suf_deg[s]] += tt.coeffs[t]
        end

        U, S, _ = svd_trunc(C; trunc = _svd_trunc(trunc))
        bond_Us[b], bond_secs[b] = _flatten_U(U, space(S, 1), preQ, pre_deg)
    end

    # 2. project each vertex operator into the compressed bases (boundary bonds are 1×1)
    r = [size(bond_Us[b], 2) for b in 1:nb]
    sizes = Tuple{Int, Int}[(b == 1 ? 1 : r[b - 1], b == N ? 1 : r[b]) for b in 1:N]
    dicts = [Dictionary{CartesianIndex{2}, LOp}() for _ in 1:N]
    fill!(cursor, 0)
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
            key = active ? tt.factors[jcur, t] :
                Factor{I}(passthrough(I), iszero(jprev) ? unit(I) : tt.factors[jprev, t].bond)
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
            lop = convert(LOp, key.letter)
            for col in 1:size(W_op, 2), row in 1:size(W_op, 1)
                iszero(W_op[row, col]) && continue
                increaseindex!(dicts[i], CartesianIndex(row, col), lop * W_op[row, col])
            end
        end
    end

    bondsectors = Vector{I}[i < N ? bond_secs[i] : I[unit(I)] for i in 1:N]
    Ws = map(sparse_from_dict, dicts, sizes)
    return (; Ws, bondsectors, starts = zeros(Int, N), finishes = zeros(Int, N))
end
