function increaseindex!(d::Dictionary, k, v)
    (found, token) = gettoken(d, k)
    if found
        settokenvalue!(d, token, gettokenvalue(d, token) + v)
    else
        insert!(d, k, v)
    end
    return d
end

mutable struct Counter <: Base.Function
    current::Int
end
Counter() = Counter(0)
(x::Counter)() = (x.current += 1; x.current)

"""
    mpo_bond_optimizations(vertices, prefix_trie) -> Vector{<:SparseMatrixDOK}

Construct MPO tensors from a prefix `Trie{Op, T}` using bipartite graph / minimum
vertex cover compression.

Each returned matrix `Ws[i]` is a `SparseMatrixDOK{LocalOp}` representing the
operator content at site `vertices[i]`.  The full Hamiltonian is recovered by the
matrix product `Ws[1] ⊗ Ws[2] ⊗ … ⊗ Ws[N]` (contracting the bond indices).

Coefficients are placed at the bond where the complementary operator is first
created.  For terms that remain in a normal (prefix-trie) state until the last
site, the coefficient is placed at the final site.
"""
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, prefix_trie::Trie{Op, T}
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

"""
    mpo_bond_optimizations(vertices, terms) -> Vector{<:SparseMatrixDOK}

Convenience overload: builds the prefix trie from `terms` then delegates to the
trie-based method.
"""
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, terms::Vector{TTNOTerm{T, Op}}
    ) where {T, Op}
    isempty(terms) && return SparseMatrixDOK{LocalOp{T, Op}}[]
    prefix_trie = Trie{Op, T}()
    for term in terms
        node = prefix_trie
        for op in term.ops
            node = get!(() -> Trie{Op, T}(), node.children, op)
        end
        node.value = something(node.value, zero(term.coeff)) + term.coeff
    end
    return mpo_bond_optimizations(vertices, prefix_trie)
end

# Convenience overload accepting a GlobalOp directly.
function mpo_bond_optimizations(vertices::AbstractVector{Int}, ex::GlobalOp{T, A}) where {T, A}
    prefix_trie = Trie{A, T}()
    build_trie!(prefix_trie, vertices, ex, one(T))
    return mpo_bond_optimizations(vertices, prefix_trie)
end
