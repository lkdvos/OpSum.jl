# ===========================================================================
# BipartiteAlgorithm on a flat TermTable
# ===========================================================================
# (`BipartiteAlgorithm` / `SVDBondAlgorithm` selectors live in src/algorithms.jl.)

# Operator at site `s` of term `t`, or identity if the term does not touch `s`.
# Relies on the columns of `tt.sites` being sorted ascending and zero-padded.
function _op_at(tt::TermTable{Op}, t::Int, s::Int) where {Op}
    @inbounds for j in 1:size(tt.sites, 1)
        st = tt.sites[j, t]
        (st == 0 || st > s) && break
        st == s && return tt.ops[j, t]
    end
    return one(Op)
end

# suf_ids[t, i] = integer class of the suffix o_t[i+1:N], assigned via
# (next_op, next_class) transitions right-to-left. Column N is the empty suffix
# (single class 1). This mirrors `SVDBondAlgorithm`'s `suf_trans` construction
# but reads ops straight off the flat table.
function _suffix_ids(tt::TermTable{Op, T}, N::Int, M::Int) where {Op, T}
    suf_ids = ones(Int, M, N)
    for i in (N - 1):-1:1
        trans = Dictionary{Tuple{Int, Op}, Int}()
        cnt = Counter()
        for t in 1:M
            key = (suf_ids[t, i + 1], _op_at(tt, t, i + 1))
            suf_ids[t, i] = get!(cnt, trans, key)
        end
    end
    return suf_ids
end

"""
    mpo_bond_optimizations(vertices, tt::TermTable, ::BipartiteAlgorithm)

Bipartite-graph / minimum-vertex-cover MPO construction driven by a flat
[`TermTable`](@ref).

Each bond-frontier state is a list of *strands* `(representative_term, coeff)`.
At each site the strands are grouped by their local operator (giving the left
`Us`) and by suffix-class transition IDs (giving the right `Vs`), and the
resulting bipartite adjacency is fed into `min_vertex_cover_bipartite`. Covered
left states are carried forward; covered right states become shared suffix
strands.
"""
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, tt::TermTable{Op, T}, ::BipartiteAlgorithm
    ) where {Op, T}
    N = length(vertices)
    M = nterms(tt)
    M == 0 && return SparseMatrixDOK{LocalOp{T, Op}}[]
    @assert nvertices(tt) == N "TermTable built for $(nvertices(tt)) vertices, got $N"

    suf_ids = _suffix_ids(tt, N, M)

    # Each frontier state is a list of strands `(representative_term, coeff)`.
    frontier = [Tuple{Int, T}[(t, tt.coeffs[t]) for t in 1:M]]

    sizes = Tuple{Int, Int}[]
    dicts = Dictionary{CartesianIndex{2}, LocalOp{T, Op}}[]

    for i in 1:N
        # --- Build the left vertices `Us` -----------------------------------
        # Each U groups one frontier state's strands sharing the same op at
        # site i. Ustrands[iu] lists (suffix_class, coeff, representative).
        Uop = Op[]
        Uleft = Int[]
        Ustrands = Vector{Tuple{Int, T, Int}}[]
        for (lid, state) in enumerate(frontier)
            groups = Dictionary{Op, Int}()
            for (repr, c) in state
                k = _op_at(tt, repr, i)
                gi = get(groups, k, 0)
                if iszero(gi)
                    push!(Uop, k)
                    push!(Uleft, lid)
                    push!(Ustrands, Tuple{Int, T, Int}[])
                    gi = length(Uop)
                    insert!(groups, k, gi)
                end
                push!(Ustrands[gi], (suf_ids[repr, i], c, repr))
            end
        end
        nU = length(Uop)

        # --- Build the right vertices `Vs`, grouped by suffix class ----------
        uid! = Counter()
        Vmap = Dictionary{Int, Int}()      # suffix class -> local column
        Vrepr = Int[]                      # representative term per column
        nonzero = Tuple{Int, Int, T}[]
        for iu in 1:nU
            for (sc, c, repr) in Ustrands[iu]
                iv = get(Vmap, sc, 0)
                if iszero(iv)
                    iv = uid!()
                    insert!(Vmap, sc, iv)
                    push!(Vrepr, repr)
                end
                push!(nonzero, (iu, iv, c))
            end
        end
        nV = uid!.current

        coefficients = zeros(T, nU, nV)
        for (iu, iv, c) in nonzero
            coefficients[iu, iv] += c
        end
        adjacency = (!iszero).(coefficients)
        coverU, coverV, _ = min_vertex_cover_bipartite(adjacency)

        mpo_terms = Pair{CartesianIndex{2}, LocalOp{T, Op}}[]
        next_frontier = Vector{Tuple{Int, T}}[]

        # covered U nodes are carried forward unchanged
        adjacency[coverU, :] .= false
        for iu in findall(coverU)
            push!(next_frontier, Tuple{Int, T}[(repr, c) for (_sc, c, repr) in Ustrands[iu]])
            j = length(next_frontier)
            k = Uop[iu]
            if i == N
                push!(mpo_terms, CartesianIndex(Uleft[iu], j) => k * coefficients[iu, 1])
            else
                push!(mpo_terms, CartesianIndex(Uleft[iu], j) => convert(LocalOp{T, Op}, k))
            end
        end

        # covered V nodes become shared suffix strands carrying coefficient 1
        for iv in findall(coverV)
            push!(next_frontier, Tuple{Int, T}[(Vrepr[iv], one(T))])
            j = length(next_frontier)
            for iu in findall(@view adjacency[:, iv])
                push!(mpo_terms, CartesianIndex(Uleft[iu], j) => Uop[iu] * coefficients[iu, iv])
            end
            adjacency[:, iv] .= false
        end
        @assert !any(adjacency)

        site_dict = Dictionary{CartesianIndex{2}, LocalOp{T, Op}}()
        for (ij, v) in mpo_terms
            increaseindex!(site_dict, ij, v)
        end
        push!(sizes, (length(frontier), length(next_frontier)))
        push!(dicts, site_dict)
        @assert i == 1 || sizes[end][1] == sizes[end - 1][2]

        frontier = next_frontier
    end

    return map(SparseArraysBase.sparse, dicts, sizes)
end

"""
    mpo_bond_optimizations(vertices, ex::GlobalOp) -> Vector{<:SparseMatrixDOK}
    mpo_bond_optimizations(vertices, ex::GlobalOp, alg) -> Vector{<:SparseMatrixDOK}

Construct MPO tensors for the operator sum `ex` over `vertices`. The optional
third argument selects the algorithm:

- `BipartiteAlgorithm()` (default): bipartite graph / minimum vertex cover.
- `SVDBondAlgorithm()`: SVD-based bond subspace selection.

Each returned `Ws[i]` is a `SparseMatrixDOK{LocalOp}` for site `vertices[i]`; the
full operator is the matrix product `Ws[1] ⊗ … ⊗ Ws[N]` over the bond indices.
The term list is held as a flat [`TermTable`](@ref); no pointer trie is built.
"""
function mpo_bond_optimizations(vertices, ex::GlobalOp{T, A}) where {T, A}
    return mpo_bond_optimizations(vertices, ex, BipartiteAlgorithm())
end

function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, ex::GlobalOp, alg::BipartiteAlgorithm
    )
    return mpo_bond_optimizations(vertices, TermTable(vertices, ex), alg)
end

function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, ex::GlobalOp, alg::SVDBondAlgorithm
    )
    return mpo_bond_optimizations(vertices, TermTable(vertices, ex), alg)
end

# ===========================================================================
# SVDBondAlgorithm
# ===========================================================================

# Shared SVD core, parameterised by accessors: `opat(t, b)` returns the operator
# at site `b` of term `t`; `coeff(t)` returns term `t`'s coefficient.
function _svd_bond_optimizations(
        ::Type{T}, ::Type{Op}, N::Int, M::Int, opat, coeff, alg::SVDBondAlgorithm
    ) where {T, Op}
    # -----------------------------------------------------------------------
    # 1. Build integer ID matrices for prefixes and suffixes at every bond.
    #    pre_ids[t, b] = unique ID for prefix ops[1:b] of term t
    #    suf_ids[t, b] = unique ID for suffix ops[b+1:N] of term t
    #
    #    IDs are assigned via (prev_id, local_op) transitions so no Vector
    #    slices need to be allocated or hashed.
    # -----------------------------------------------------------------------
    pre_ids = zeros(Int, M, N - 1)
    suf_ids = zeros(Int, M, N - 1)

    pre_trans = [Dictionary{Tuple{Int, Op}, Int}() for _ in 1:(N - 1)]
    pre_counters = [Counter() for _ in 1:(N - 1)]
    for t in 1:M
        prev_id = 1   # sentinel for empty prefix
        for b in 1:(N - 1)
            id = get!(pre_counters[b], pre_trans[b], (prev_id, opat(t, b)))
            pre_ids[t, b] = id
            prev_id = id
        end
    end

    suf_trans = [Dictionary{Tuple{Int, Op}, Int}() for _ in 1:(N - 1)]
    suf_counters = [Counter() for _ in 1:(N - 1)]
    for t in 1:M
        prev_id = 1   # sentinel for empty suffix
        for b in (N - 1):-1:1
            id = get!(suf_counters[b], suf_trans[b], (prev_id, opat(t, b + 1)))
            suf_ids[t, b] = id
            prev_id = id
        end
    end

    # -----------------------------------------------------------------------
    # 2. Assemble coefficient matrices C[b] (n_pre × n_suf)
    # -----------------------------------------------------------------------
    Cs = [zeros(T, pre_counters[b].current, suf_counters[b].current) for b in 1:(N - 1)]
    for t in 1:M
        for b in 1:(N - 1)
            Cs[b][pre_ids[t, b], suf_ids[t, b]] += coeff(t)
        end
    end

    # -----------------------------------------------------------------------
    # 3. SVD each bond — keep only left singular vectors U[b] (n_pre × r)
    # -----------------------------------------------------------------------
    default_trunc = trunctol(rtol = eps(real(T)))
    trunc = something(alg.trunc, default_trunc)

    bond_Us = Vector{Matrix{T}}(undef, N - 1)
    for b in 1:(N - 1)
        U, _, _ = svd_trunc!(Cs[b]; trunc)
        bond_Us[b] = U    # shape (n_pre_b × r_b)
    end

    # -----------------------------------------------------------------------
    # 4. Build per-operator matrices in the (pre_{i-1}, pre_i) basis,
    #    then compress: W_op = U[i-1]ᵀ · C_op · U[i]
    # -----------------------------------------------------------------------
    r = [size(bond_Us[b], 2) for b in 1:(N - 1)]
    sizes = [(b == 1 ? 1 : r[b - 1], b == N ? 1 : r[b]) for b in 1:N]
    dicts = [Dictionary{CartesianIndex{2}, LocalOp{T, Op}}() for _ in 1:N]

    for i in 1:N
        U_left = i > 1 ? bond_Us[i - 1] : ones(T, 1, 1)
        U_right = i < N ? bond_Us[i] : ones(T, 1, 1)
        n_pre_left = size(U_left, 1)
        n_pre_right = size(U_right, 1)

        op_coeffs = Dictionary{Op, Matrix{T}}()
        for t in 1:M
            op = opat(t, i)
            j = i > 1 ? pre_ids[t, i - 1] : 1
            l = i < N ? pre_ids[t, i] : 1
            C = get!(() -> zeros(T, n_pre_left, n_pre_right), op_coeffs, op)
            if i == N
                C[j, l] += coeff(t)         # accumulate: multiple terms can share the same prefix
            else
                C[j, l] = one(T)            # deterministic: same (j,l) implies same op
            end
        end

        for (op, C_op) in pairs(op_coeffs)
            W_op = U_left' * C_op * U_right   # shape (r_left × r_right)
            local_op = convert(LocalOp{T, Op}, op)
            for col in 1:size(W_op, 2), row in 1:size(W_op, 1)
                iszero(W_op[row, col]) && continue
                increaseindex!(dicts[i], CartesianIndex(row, col), local_op * W_op[row, col])
            end
        end
    end

    return map(SparseArraysBase.sparse, dicts, sizes)
end

"""
    mpo_bond_optimizations(vertices, tt::TermTable, alg::SVDBondAlgorithm)

SVD-based MPO construction.

At each bond b, assemble the coefficient matrix C[b][pre, suf] (shape n_pre × n_suf)
from all operator terms.  The left singular vectors U[b] (n_pre × r) define the
compressed bond basis of rank r.

Vertex operators are first assembled in the uncompressed (pre_{i-1}, pre_i) basis,
then projected:

    W_compressed[i] = U[i-1]ᵀ · W_uncompressed[i] · U[i]

where U[0] = U[N] = I₁ at the chain boundaries.
"""
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, tt::TermTable{Op, T}, alg::SVDBondAlgorithm
    ) where {T, Op}
    N = length(vertices)
    M = nterms(tt)
    M == 0 && return SparseMatrixDOK{LocalOp{T, Op}}[]
    @assert nvertices(tt) == N "TermTable built for $(nvertices(tt)) vertices, got $N"
    return _svd_bond_optimizations(
        T, Op, N, M, (t, b) -> _op_at(tt, t, b), t -> tt.coeffs[t], alg
    )
end
