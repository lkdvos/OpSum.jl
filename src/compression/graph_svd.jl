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

# SVD a charge-graded block `C : Ppre ← Psuf` (truncated by `trunc`) and flatten the
# retained `U` into a dense `Umat[p, col]` with column charges `secs`. With `colQ`/`coldeg`/`colactive`
# it also flattens `R = S · Vᴴ` into `Rmat` (what `SequentialSVD` forwards).
function _svd_flatten_block(
        C, trunc, rowQ::Vector{I}, rowdeg::Vector{Int};
        colQ::Union{Nothing, Vector{I}} = nothing, coldeg::Union{Nothing, Vector{Int}} = nothing,
        colactive::Union{Nothing, BitVector} = nothing
    ) where {I}
    U, S, Vt = svd_trunc(C; trunc = _svd_trunc(trunc))
    Wb = space(S, 1)   # retained bond space
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
    col == r || invariant_violation("retained bond space does not match its per-sector dimensions")
    return (; U = Umat, secs, R = Rmat)
end

# `bond_basis!` for `SequentialSVD`: the whole bond's coefficient matrix is a charge-graded
# `C : Ppre ← Psuf`; its truncated `U` is the compressed basis (index `m` emits `key.letter * U[u, m]`) and
# `R = S·Vᴴ` becomes the forwarded edge weights (folded into the block at the last site).
function bond_basis!(g::SweepGraph{I}, i::Int, nU::Int, nV::Int, strategy::SequentialSVD) where {I}
    LOp = SiteOperator{I}
    N = g.N
    coeff = zeros(ComplexF64, nU, nV)   # dense `coeff[u, v]` from the sparse adjacency
    for iu in 1:nU
        for (rid, w) in zip(g.left.radj[iu], g.left.wadj[iu])
            coeff[iu, rid] += w
        end
    end

    # bond charge of each prefix and suffix state (a right vertex orphaned by truncation has none)
    uCharge = I[g.left.vertex[u].key.bond for u in 1:nU]
    vCharge = Vector{I}(undef, nV)
    vactive = falses(nV)
    for v in 1:nV
        conn = findall(!iszero, @view coeff[:, v])
        isempty(conn) && continue
        q = vCharge[v] = uCharge[first(conn)]
        all(uCharge[u] == q for u in conn) ||
            invariant_violation("bond index not sector-pure (block-diagonality violated)")
        vactive[v] = true
    end

    # degeneracy index within the charge sector, and per-sector multiplicities
    umult, vmult = Dict{I, Int}(), Dict{I, Int}()
    udeg = [_nextdeg!(umult, q) for q in uCharge]
    vdeg = [vactive[v] ? _nextdeg!(vmult, vCharge[v]) : 0 for v in 1:nV]

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

        flat = _svd_flatten_block(C, trunc, preQ, pre_deg)
        bond_Us[b] = flat.U
        bond_secs[b] = flat.secs
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
