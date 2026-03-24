"""Algorithm selector: bipartite graph / minimum vertex cover (current default)."""
struct BipartiteAlgorithm end

"""Algorithm selector: SVD-based bond subspace selection."""
struct SVDBondAlgorithm
    trunc  # TruncationStrategy or nothing
end
SVDBondAlgorithm() = SVDBondAlgorithm(nothing)

"""
    mpo_bond_optimizations(vertices, prefix_trie) -> Vector{<:SparseMatrixDOK}
    mpo_bond_optimizations(vertices, prefix_trie, alg) -> Vector{<:SparseMatrixDOK}

Construct MPO tensors from a prefix `Trie{Op, T}`. The optional third argument
selects the algorithm:

- `BipartiteAlgorithm()` (default): bipartite graph / minimum vertex cover.
- `SVDBondAlgorithm()`: SVD-based bond subspace selection.

Each returned matrix `Ws[i]` is a `SparseMatrixDOK{LocalOp}` representing the
operator content at site `vertices[i]`.  The full Hamiltonian is recovered by the
matrix product `Ws[1] ⊗ Ws[2] ⊗ … ⊗ Ws[N]` (contracting the bond indices).
"""
function mpo_bond_optimizations(vertices, prefix_trie::Trie{Op, T}) where {T, Op}
    return mpo_bond_optimizations(vertices, prefix_trie, BipartiteAlgorithm())
end

function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, prefix_trie::Trie{Op, T}, ::BipartiteAlgorithm
    ) where {T, Op}
    N = length(vertices)
    isempty(prefix_trie) && return SparseMatrixDOK{LocalOp{T, Op}}[]

    # -----------------------------------------------------------------------
    # Initialise sweep state
    # -----------------------------------------------------------------------
    W = Trie{Op, T}[]
    left_nodes = Trie{Op, T}[prefix_trie]
    sizes = Tuple{Int, Int}[]
    dicts = Dictionary{CartesianIndex{2}, LocalOp{T, Op}}[]

    for i in 1:N
        Us = Trie{Op, T}[]
        parents = Pair{Trie{Op, T}, Op}[]
        uidx_ = Int[]
        uidx__ = 0
        mpo_terms = Pair{CartesianIndex{2}, LocalOp{T, Op}}[]
        @debug "starting from" left_nodes
        for node in left_nodes
            uidx__ += 1
            for (k, child) in pairs(node.children)
                push!(Us, child)
                push!(parents, node => k)
                push!(uidx_, uidx__)
            end
        end

        uid! = Counter()
        Vs = Dictionary{Vector{Op}, Int}()
        nonzero_list = Pair{CartesianIndex{2}, T}[]
        for (iu, U) in enumerate(Us)
            if isempty(U)
                iv = get!(uid!, Vs, keytype(U)[])
                push!(nonzero_list, CartesianIndex(iu, iv) => something(U.value))
            else
                for (operator, coeff) in pairs(U)
                    iv = get!(uid!, Vs, operator)
                    push!(nonzero_list, CartesianIndex(iu, iv) => coeff)
                end
            end
        end

        coefficients = zeros(T, length(Us), uid!.current)
        for (I, c) in nonzero_list
            @assert iszero(coefficients[Tuple(I)...])
            coefficients[Tuple(I)...] = c
        end
        adjacency = (!iszero).(coefficients)
        coverU, coverV, _ = min_vertex_cover_bipartite(adjacency)

        @debug "adjacency at site $i" Us Vs coefficients
        @debug "covering at site $i" adjacency coverU coverV

        # cover U nodes are simply passed through
        W = Us[coverU]
        adjacency[coverU, :] .= false
        uidnext! = Counter()
        Wnext_dict = IdDict{Trie{Op, T}, Int}()

        for iu in findall(coverU)
            left_id = uidx_[iu]
            (node, k) = parents[iu]
            j = get!(uidnext!, Wnext_dict, Us[iu])
            if i == N
                c = coefficients[iu, 1]
                push!(mpo_terms, CartesianIndex(left_id, j) => k * c)
            else
                push!(mpo_terms, CartesianIndex(left_id, j) => convert(LocalOp{T, Op}, k))
            end
        end

        # cover V nodes need special handling since we need to create new nodes and correctly connect them
        Vkeys = collect(keys(Vs))
        for iv in findall(coverV)
            suffix = Vkeys[iv]
            newnode = typeof(prefix_trie)()
            push!(W, newnode)
            node = newnode
            for s in suffix
                newnode = typeof(prefix_trie)()
                insert!(node.children, s, newnode)
                node = newnode
            end
            node.value = one(T)

            j = uidnext!()
            for iu in findall(adjacency[:, iv])
                left_id = uidx_[iu]
                node, k = parents[iu]
                c = coefficients[iu, iv]
                push!(mpo_terms, CartesianIndex(left_id, j) => k * c)
            end
            adjacency[:, iv] .= false
        end
        @assert !any(adjacency)

        site_dict = Dictionary{CartesianIndex{2}, LocalOp{T, Op}}()
        for (ij, k) in mpo_terms
            increaseindex!(site_dict, ij, k)
        end
        push!(sizes, (length(left_nodes), length(W)))
        push!(dicts, site_dict)

        @assert i == 1 || sizes[end][1] == sizes[end - 1][2]

        left_nodes = W
    end

    return map(SparseArraysBase.sparse, dicts, sizes)
end

function mpo_bond_optimizations(vertices, terms::Vector{TTNOTerm{T, Op}}) where {T, Op}
    return mpo_bond_optimizations(vertices, terms, BipartiteAlgorithm())
end

function mpo_bond_optimizations(vertices, ex::GlobalOp{T, A}) where {T, A}
    return mpo_bond_optimizations(vertices, ex, BipartiteAlgorithm())
end

"""
    mpo_bond_optimizations(vertices, terms, alg) -> Vector{<:SparseMatrixDOK}

Convenience overload: builds the prefix trie from `terms` then delegates to the
trie-based method with the given algorithm.
"""
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, terms::Vector{TTNOTerm{T, Op}}, alg
    ) where {T, Op}
    isempty(terms) && return SparseMatrixDOK{LocalOp{T, Op}}[]
    return mpo_bond_optimizations(vertices, _build_prefix_trie(terms), alg)
end

# Convenience overload accepting a GlobalOp directly.
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, ex::GlobalOp{T, A}, alg
    ) where {T, A}
    prefix_trie = Trie{A, T}()
    build_trie!(prefix_trie, vertices, ex, one(T))
    return mpo_bond_optimizations(vertices, prefix_trie, alg)
end

function _build_prefix_trie(terms::Vector{TTNOTerm{T, Op}}) where {T, Op}
    prefix_trie = Trie{Op, T}()
    for term in terms
        node = prefix_trie
        for op in term.ops
            node = get!(() -> Trie{Op, T}(), node.children, op)
        end
        node.value = something(node.value, zero(term.coeff)) + term.coeff
    end
    return prefix_trie
end

# ===========================================================================
# SVDBondAlgorithm
# ===========================================================================

"""
    mpo_bond_optimizations(vertices, prefix_trie, alg::SVDBondAlgorithm)

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
        vertices::AbstractVector{Int}, prefix_trie::Trie{Op, T}, alg::SVDBondAlgorithm
    ) where {T, Op}
    N = length(vertices)
    isempty(prefix_trie) && return SparseMatrixDOK{LocalOp{T, Op}}[]

    # -----------------------------------------------------------------------
    # 1. Enumerate unique prefixes and suffixes at every bond
    # -----------------------------------------------------------------------
    bond_pre_maps = [Dictionary{Vector{Op}, Int}() for _ in 1:(N - 1)]
    bond_suf_maps = [Dictionary{Vector{Op}, Int}() for _ in 1:(N - 1)]

    for (ops, _) in pairs(prefix_trie)
        for b in 1:(N - 1)
            get!(bond_pre_maps[b], ops[1:b], length(bond_pre_maps[b]) + 1)
            get!(bond_suf_maps[b], ops[(b + 1):end], length(bond_suf_maps[b]) + 1)
        end
    end

    # -----------------------------------------------------------------------
    # 2. Assemble coefficient matrices C[b] (n_pre × n_suf)
    # -----------------------------------------------------------------------
    Cs = [zeros(T, length(bond_pre_maps[b]), length(bond_suf_maps[b])) for b in 1:(N - 1)]
    for (ops, coeff) in pairs(prefix_trie)
        for b in 1:(N - 1)
            Cs[b][bond_pre_maps[b][ops[1:b]], bond_suf_maps[b][ops[(b + 1):end]]] += coeff
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
        pre_left_map = i > 1 ? bond_pre_maps[i - 1] : Dictionary([Op[]], [1])
        pre_right_map = i < N ? bond_pre_maps[i] : Dictionary([Op[]], [1])
        n_pre_left = size(U_left, 1)
        n_pre_right = size(U_right, 1)

        op_coeffs = Dictionary{Op, Matrix{T}}()
        for (ops, coeff) in pairs(prefix_trie)
            op = ops[i]
            pre_left = i > 1 ? ops[1:(i - 1)] : Op[]
            pre_right = i < N ? ops[1:i] : Op[]
            j = pre_left_map[pre_left]
            l = pre_right_map[pre_right]
            C = get!(() -> zeros(T, n_pre_left, n_pre_right), op_coeffs, op)
            if i == N
                C[j, l] += coeff    # accumulate: multiple terms can share the same prefix
            else
                C[j, l] = one(T)    # deterministic: same (j,l) implies same op
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
