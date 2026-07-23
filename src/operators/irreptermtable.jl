# Flat term-list storage + per-bond-sector bipartite sweep for the ITO automaton
# ==============================================================================
# `ITOTermTable` stores each term's active `(site, ITOKey)` factors in flat matrices (the ITO
# counterpart of the dense `TermTable`); `_irrep_bipartite` runs the per-bond-sector min-vertex-
# cover sweep directly off it. `irrep_mpo` (irrepmpo.jl) is the public entry.
#
# The one ITO-specific subtlety vs the dense `TermTable`: idle sites are NOT a constant identity —
# they carry a pass-through symbol with the *running* bond charge. The flat store keeps only active
# factors; `_op_at_ito` reconstructs the pass-through key at any idle position from the last active
# bond charge to its left, so the per-bond transition key is effectively `(prev_id, op, bond,
# vertex)`, preserving block-diagonality exactly.
#
# The sweep uses sorted `Us` and materialised-suffix `Vs` keys, feeding `min_vertex_cover_bipartite`
# a canonical bipartite graph and yielding the per-sector bond dimensions the ITO tests pin down.

using TensorKit: Sector, unit, Vect, block, sectors, space, dim
using MatrixAlgebraKit: svd_trunc, trunctol

"""
    ITOTermTable{I<:Sector}

Flat, sparse-per-term storage of an ITO term-sum on an `N`-vertex chain: each term's active
`(site, ITOKey)` factors in `K×M` matrices (`sites` zero-padded, ascending) plus a parallel
`coeffs` vector. The ITO counterpart of the dense [`TermTable`](@ref).
"""
struct ITOTermTable{I <: Sector}
    sites::Matrix{Int}
    keys::Matrix{ITOKey{I}}
    coeffs::Vector{ComplexF64}
    nvertices::Int
end

arity(tt::ITOTermTable) = size(tt.sites, 1)
nterms(tt::ITOTermTable) = length(tt.coeffs)
nvertices(tt::ITOTermTable) = tt.nvertices

# Active `(site => ITOKey)` factors of a term, in site order (empty for a K=0 identity term).
function _active_keys(tk::TermKey{I, S}) where {I, S}
    K = length(tk.sites)
    K == 0 && return Pair{Int, ITOKey{I}}[]
    bonds = bondcharges(tk.tree)
    verts = vertexlabels(tk.tree)
    return Pair{Int, ITOKey{I}}[Int(tk.sites[a]) => ITOKey{I}(tk.ops[a], bonds[a], verts[a]) for a in 1:K]
end

"""
    ITOTermTable(ts::TermSum, sites)
    ITOTermTable(terms, sites)      # iterable of TermSums, summed first

Build the flat ITO term table for a term-sum over `N = length(sites)` lattice sites. Coincident
paths (identical active `(site, ITOKey)` content) accumulate coefficients.
"""
function ITOTermTable(ts::TermSum{I, S, Tc}, sites) where {I, S, Tc}
    N = length(sites)
    index = Dictionary{Vector{Pair{Int, ITOKey{I}}}, Int}()
    keys_ = Vector{Pair{Int, ITOKey{I}}}[]
    coeffs = ComplexF64[]
    for (termkey, coeff) in pairs(ts.terms)
        active = _active_keys(termkey)
        tok = get(index, active, 0)
        if iszero(tok)
            push!(keys_, active)
            push!(coeffs, ComplexF64(coeff))
            insert!(index, active, length(keys_))
        else
            coeffs[tok] += ComplexF64(coeff)
        end
    end

    M = length(keys_)
    K = max(1, maximum(length, keys_; init = 0))
    sitemat = zeros(Int, K, M)
    keymat = fill(ITOKey{I}(passthrough(I), unit(I), 1), K, M)
    for t in 1:M
        for (j, (s, key)) in enumerate(keys_[t])
            sitemat[j, t] = s
            keymat[j, t] = key
        end
    end
    return ITOTermTable{I}(sitemat, keymat, coeffs, N)
end

ITOTermTable(terms::AbstractVector{<:TermSum}, sites) = ITOTermTable(reduce(+, terms), sites)

# ITOKey at site `s` of term `t`. On an active site it is the stored key; on an idle site it is the
# pass-through symbol carrying the running bond charge (the last active bond to the left of `s`, or
# `unit(I)` if none). Relies on `sites` columns being sorted ascending and zero-padded.
function _op_at_ito(tt::ITOTermTable{I}, t::Int, s::Int) where {I}
    lastbond = unit(I)
    @inbounds for j in 1:size(tt.sites, 1)
        st = tt.sites[j, t]
        st == 0 && break
        if st == s
            return tt.keys[j, t]
        elseif st < s
            lastbond = tt.keys[j, t].bond
        else
            break
        end
    end
    return ITOKey{I}(passthrough(I), lastbond, 1)
end

# Materialised suffix path o_t[i+1 .. N] of term `t` (the `Vs` grouping key).
function _suffix_path(tt::ITOTermTable{I}, t::Int, i::Int, N::Int) where {I}
    return ITOKey{I}[_op_at_ito(tt, t, s) for s in (i + 1):N]
end

# Per-bond-sector bipartite min-vertex-cover sweep over the flat `ITOTermTable`. Each frontier state
# is a list of strands `(representative_term, coeff)`; `Us` are the per-state groups by local ITOKey
# (sorted), `Vs` are grouped by materialised suffix path. Emits ITO letters (`ITOKey.op`) times
# reduced coefficients and tracks each retained bond's charge (`ITOKey.bond`), asserting sector
# purity. `irrep_mpo` (irrepmpo.jl) is the public entry.
function _irrep_bipartite(tt::ITOTermTable{I}, N::Int) where {I}
    Op = ITOKey{I}
    T = ComplexF64
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    M = nterms(tt)
    M == 0 && return (SparseMatrixDOK{LOp}[], Vector{I}[])

    Strand = Tuple{Int, T}                # (representative term, coeff)
    frontier = [Tuple{Int, T}[(t, tt.coeffs[t]) for t in 1:M]]

    sizes = Tuple{Int, Int}[]
    dicts = Dictionary{CartesianIndex{2}, LOp}[]
    bondsectors = Vector{I}[]

    for i in 1:N
        # --- Us: per-state groups by local ITOKey, sorted -------------------------------------
        Uop = Op[]
        Uleft = Int[]
        Ustrands = Vector{Strand}[]
        for (lid, state) in enumerate(frontier)
            groups = Dictionary{Op, Vector{Strand}}()
            for str in state
                k = _op_at_ito(tt, str[1], i)
                push!(get!(() -> Strand[], groups, k), str)
            end
            for k in sort!(collect(keys(groups)))
                push!(Uop, k)
                push!(Uleft, lid)
                push!(Ustrands, groups[k])
            end
        end
        nU = length(Uop)

        # --- Vs: grouped by materialised suffix path ------------------------------------------
        uid! = Counter()
        Vs = Dictionary{Vector{Op}, Int}()
        Vrepr = Int[]
        nonzero_list = Pair{CartesianIndex{2}, T}[]
        for iu in 1:nU
            strands = Ustrands[iu]
            suffixes = [(_suffix_path(tt, str[1], i, N), str) for str in strands]
            sort!(suffixes; by = first)
            for (suffix, str) in suffixes
                iv = get(Vs, suffix, 0)
                if iszero(iv)
                    iv = uid!()
                    insert!(Vs, suffix, iv)
                    push!(Vrepr, str[1])
                end
                push!(nonzero_list, CartesianIndex(iu, iv) => str[2])
            end
        end
        nV = uid!.current

        coefficients = zeros(T, nU, nV)
        adjU = [Int[] for _ in 1:nU]
        for (Idx, c) in nonzero_list
            iu, iv = Tuple(Idx)
            @assert iszero(coefficients[iu, iv])
            coefficients[iu, iv] = c
            push!(adjU[iu], iv)
        end
        adjacency = (!iszero).(coefficients)

        # Split into connected components and run the min-vertex-cover per component instead of
        # once over the whole (block-diagonal-by-bond-charge) site graph. A minimum vertex cover of
        # a disjoint union of graphs is exactly the union of the components' minimum vertex covers
        # (König's theorem applies per-component), so this is a pure decomposition: same cover, same
        # bond dimension, just smaller independent matching problems.
        coverU = falses(nU)
        coverV = falses(nV)
        us_of_component, vs_of_component = bipartite_connected_components(adjU, nV)
        for (us, vs) in zip(us_of_component, vs_of_component)
            cU, cV, _ = min_vertex_cover_bipartite(adjacency[us, vs])
            coverU[us[cU]] .= true
            coverV[vs[cV]] .= true
        end

        ubond = I[Uop[iu].bond for iu in 1:nU]

        mpo_terms = Pair{CartesianIndex{2}, LOp}[]
        next_frontier = Vector{Strand}[]
        secW = I[]

        adjacency[coverU, :] .= false
        for iu in findall(coverU)
            push!(next_frontier, Strand[str for str in Ustrands[iu]])
            j = length(next_frontier)
            push!(secW, ubond[iu])
            k = Uop[iu]
            if i == N
                push!(mpo_terms, CartesianIndex(Uleft[iu], j) => k.op * coefficients[iu, 1])
            else
                push!(mpo_terms, CartesianIndex(Uleft[iu], j) => convert(LOp, k.op))
            end
        end

        for iv in findall(coverV)
            push!(next_frontier, Strand[(Vrepr[iv], one(T))])
            j = length(next_frontier)

            # Now implied by connectivity (all of `conn` shares iv's connected component, and the
            # automaton is block-diagonal in the bond charge — irrepkey.jl:6), kept as a cheap
            # per-component safety net rather than removed.
            conn = findall(!iszero, coefficients[:, iv])
            charge = ubond[first(conn)]
            @assert all(ubond[iu] == charge for iu in conn) "bond index not sector-pure (block-diagonality violated)"
            push!(secW, charge)

            for iu in findall(@view adjacency[:, iv])
                push!(mpo_terms, CartesianIndex(Uleft[iu], j) => Uop[iu].op * coefficients[iu, iv])
            end
            adjacency[:, iv] .= false
        end
        @assert !any(adjacency)
        @assert length(secW) == length(next_frontier)

        site_dict = Dictionary{CartesianIndex{2}, LOp}()
        for (ij, k) in mpo_terms
            increaseindex!(site_dict, ij, k)
        end
        push!(sizes, (length(frontier), length(next_frontier)))
        push!(dicts, site_dict)
        push!(bondsectors, secW)
        @assert i == 1 || sizes[end][1] == sizes[end - 1][2]

        frontier = next_frontier
    end

    return (map(SparseArraysBase.sparse, dicts, sizes), bondsectors)
end

# Per-bond-sector SVD sweep over the flat `ITOTermTable` — the ITO counterpart of the dense
# `_svd_bond_optimizations` (graphbuilding.jl). Same skeleton (prefix/suffix transition IDs, one
# coefficient matrix per bond, keep the left singular vectors as the compressed bond basis, then
# project the vertex operators), with one ITO-specific change: each bond coefficient matrix is built
# as a *charge-graded* `TensorMap C_b : Ppre ← Psuf` — both spaces graded by the running bond charge
# `_op_at_ito(tt,t,b).bond`. That makes `C_b` block-diagonal in the bond charge, so `svd_trunc`
# does the per-sector SVD *and* the global-across-sectors truncation automatically (respecting the
# quantum dimensions), and the retained bond space directly gives `bondsectors`. The compression is
# on the symbolic bond coefficients only — entries stay ITO letters times scalars — so the output
# `(Ws, bondsectors)` feeds `irrep_mpo_tensors` unchanged. `trunc === nothing` ⇒ lossless default.
function _irrep_svd(tt::ITOTermTable{I}, N::Int, trunc) where {I}
    Op = ITOKey{I}
    T = ComplexF64
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    M = nterms(tt)
    M == 0 && return (SparseMatrixDOK{LOp}[], Vector{I}[])

    _bondcharge(t, b) = _op_at_ito(tt, t, b).bond

    # --- 1. Prefix / suffix transition IDs at every internal bond -----------------------------
    #   pre_ids[t, b] = class of prefix o_t[1:b];  suf_ids[t, b] = class of suffix o_t[b+1:N].
    #   Assigned via (prev_id, ITOKey) transitions (the ITOKey carries op+bond+vertex, so classes
    #   are automatically sector-pure). Mirrors graphbuilding.jl but reads keys via `_op_at_ito`.
    pre_ids = zeros(Int, M, max(N - 1, 0))
    suf_ids = zeros(Int, M, max(N - 1, 0))
    pre_trans = [Dictionary{Tuple{Int, Op}, Int}() for _ in 1:(N - 1)]
    pre_cnt = [Counter() for _ in 1:(N - 1)]
    for t in 1:M
        prev = 1
        for b in 1:(N - 1)
            id = get!(pre_cnt[b], pre_trans[b], (prev, _op_at_ito(tt, t, b)))
            pre_ids[t, b] = id
            prev = id
        end
    end
    suf_trans = [Dictionary{Tuple{Int, Op}, Int}() for _ in 1:(N - 1)]
    suf_cnt = [Counter() for _ in 1:(N - 1)]
    for t in 1:M
        prev = 1
        for b in (N - 1):-1:1
            id = get!(suf_cnt[b], suf_trans[b], (prev, _op_at_ito(tt, t, b + 1)))
            suf_ids[t, b] = id
            prev = id
        end
    end

    # --- 2. Per bond: build the charge-graded coefficient TensorMap, SVD-truncate, keep U -----
    #   `bond_Us[b]` is the (n_pre × r_b) left-isometry as a plain matrix (block-diagonal in charge,
    #   columns grouped per sector); `bond_secs[b]` is the retained bond's charge per column.
    bond_Us = Vector{Matrix{T}}(undef, N - 1)
    bond_secs = Vector{Vector{I}}(undef, N - 1)
    truncstrat = something(trunc, trunctol(rtol = eps(Float64)))
    for b in 1:(N - 1)
        npre = pre_cnt[b].current
        nsuf = suf_cnt[b].current
        # charge of each prefix / suffix class (the shared bond charge); set-once, assert-consistent
        preQ = Vector{Union{Nothing, I}}(nothing, npre)
        sufQ = Vector{Union{Nothing, I}}(nothing, nsuf)
        for t in 1:M
            q = _bondcharge(t, b)
            p, s = pre_ids[t, b], suf_ids[t, b]
            @assert preQ[p] === nothing || preQ[p] == q "prefix class not sector-pure"
            @assert sufQ[s] === nothing || sufQ[s] == q "suffix class not sector-pure"
            preQ[p] = q
            sufQ[s] = q
        end
        # degeneracy of each class within its charge sector + per-sector multiplicities
        pre_mult = Dict{I, Int}()
        pre_deg = zeros(Int, npre)
        for p in 1:npre
            q = something(preQ[p])
            pre_deg[p] = pre_mult[q] = get(pre_mult, q, 0) + 1
        end
        suf_mult = Dict{I, Int}()
        suf_deg = zeros(Int, nsuf)
        for s in 1:nsuf
            q = something(sufQ[s])
            suf_deg[s] = suf_mult[q] = get(suf_mult, q, 0) + 1
        end

        Ppre = Vect[I](pre_mult)
        Psuf = Vect[I](suf_mult)
        C = zeros(T, Ppre ← Psuf)
        for t in 1:M
            q = _bondcharge(t, b)
            block(C, q)[pre_deg[pre_ids[t, b]], suf_deg[suf_ids[t, b]]] += tt.coeffs[t]
        end

        U, S, _ = svd_trunc(C; trunc = truncstrat)
        Wb = space(S, 1)   # retained bond space (⊕ charge sectors with truncated multiplicities)

        # flatten U into an (npre × r_b) block-diagonal matrix, columns grouped per sector
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
                    something(preQ[p]) == q || continue
                    Umat[p, col] = Ub[pre_deg[p], dcol]
                end
            end
        end
        @assert col == r_b
        bond_Us[b] = Umat
        bond_secs[b] = secs
    end

    # --- 3. Project each vertex operator into the compressed bond bases -----------------------
    #   W_op = U_{i-1}' · C_op · U_i, per ITO letter, in the uncompressed (pre_{i-1}, pre_i) basis
    #   (boundary bonds are the 1×1 identity). Emits the letter times the compressed coefficient.
    r = [size(bond_Us[b], 2) for b in 1:(N - 1)]
    sizes = Tuple{Int, Int}[(b == 1 ? 1 : r[b - 1], b == N ? 1 : r[b]) for b in 1:N]
    dicts = [Dictionary{CartesianIndex{2}, LOp}() for _ in 1:N]
    for i in 1:N
        U_left = i > 1 ? bond_Us[i - 1] : ones(T, 1, 1)
        U_right = i < N ? bond_Us[i] : ones(T, 1, 1)
        nL = size(U_left, 1)
        nR = size(U_right, 1)

        op_coeffs = Dictionary{Op, Matrix{T}}()
        for t in 1:M
            key = _op_at_ito(tt, t, i)
            j = i > 1 ? pre_ids[t, i - 1] : 1
            l = i < N ? pre_ids[t, i] : 1
            Cmat = get!(() -> zeros(T, nL, nR), op_coeffs, key)
            if i == N
                Cmat[j, l] += tt.coeffs[t]   # accumulate: many terms can share the same prefix
            else
                Cmat[j, l] = one(T)          # deterministic: same (j,l,key) ⇒ same successor
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
    return (map(SparseArraysBase.sparse, dicts, sizes), bondsectors)
end
