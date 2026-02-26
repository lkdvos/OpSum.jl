struct NormalLeftNode
    uid::Int
end

struct ComplementaryLeftNode
    uid::Int
end

const LeftNode = Union{NormalLeftNode, ComplementaryLeftNode}

function trie_hash!(hashnode::Trie{K, UInt})::UInt where {K}
    if isnothing(hashnode.value)
        children_hash = map(trie_hash!, hashnode.children)
        sortkeys!(children_hash)
        hashnode.value = hash(children_hash)
    end
    return hashnode.value
end

mutable struct Counter <: Base.Function
    current::Int
end
Counter() = Counter(0)
(x::Counter)() = (x.current += 1; x.current)

"""
    mpo_bond_optimizations(vertices, terms) -> Vector{<:SparseMatrixDOK}

Construct MPO tensors from a list of `TTNOTerm`s using bipartite graph / minimum
vertex cover compression.

Each returned matrix `Ws[i]` is a `SparseMatrixDOK{LocalOp}` representing the
operator content at site `vertices[i]`.  The full Hamiltonian is recovered by the
matrix product `Ws[1] ⊗ Ws[2] ⊗ … ⊗ Ws[N]` (contracting the bond indices).

Coefficients are placed at the bond where the complementary operator is first
created.  For terms that remain in a normal (prefix-trie) state until the last
site, the coefficient is placed at the final site.
"""
function mpo_bond_optimizations(
        vertices::AbstractVector{Int}, terms::Vector{TTNOTerm{T, Op}}
    ) where {T, Op}
    N = length(vertices)
    K = length(terms)
    K == 0 && return SparseMatrixDOK{LocalOp{T, Op}}[]

    coefficients = map(x -> x.coeff, terms)

    # -----------------------------------------------------------------------
    # Step 1: Build prefix trie
    # -----------------------------------------------------------------------
    prefix_trie = Trie{Op, T}()
    for o in 1:K
        node = prefix_trie
        for op in terms[o].ops
            node = get!(() -> typeof(node)(), node.children, op)
        end
        node.value = terms[o].coeff
    end
    # populate hash values
    # trie_hash!(prefix_trie)

    # -----------------------------------------------------------------------
    # Step 2: Initialise sweep state
    # -----------------------------------------------------------------------
    Ws = []
    W = []
    left_nodes = [prefix_trie]
    @debug "transition matrix" W
    mpos = SparseMatrixDOK{LocalOp{T, Op}}[]

    for i in 1:N
        Us = []
        parents = []
        uidx_ = []
        uidx__ = 0
        mpo_terms = []
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
        Vs = Dictionary()
        nonzero_list = CartesianIndex{2}[]
        for (iu, U) in enumerate(Us)
            if isempty(U)
                iv = get!(uid!, Vs, keytype(U)[])
                push!(nonzero_list, CartesianIndex(iu, iv))
            else
                for operator in collect(keys(U))
                    iv = get!(uid!, Vs, operator)
                    push!(nonzero_list, CartesianIndex(iu, iv))
                end
            end
        end

        adjacency = falses(length(Us), uid!.current)
        for I in nonzero_list
            adjacency[Tuple(I)...] = true
        end

        coverU, coverV, _ = min_vertex_cover_bipartite(adjacency)

        @debug "adjacency at site $i" Us Vs adjacency coverU coverV

        # cover U nodes are simply passed through
        W = Us[coverU]
        push!(Ws, W)
        adjacency[coverU, :] .= false
        uidnext! = Counter()
        Wnext_dict = Dictionary()

        for iu in findall(coverU)
            left_id = uidx_[iu]
            @show (node, k) = parents[iu]
            j = get!(uidnext!, Wnext_dict, Us[iu])
            push!(mpo_terms, (left_id, j) => k)
        end

        # # but disconnect uncovered nodes
        # for iu in findall(!, coverU)
        #     node, k = parents[iu]
        #     delete!(node.children, k)
        # end

        @debug "here W is" W mpo_terms

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

            for iu in findall(adjacency[:, iv])
                left_id = uidx_[iu]
                node, k = parents[iu]
                j = get!(uidnext!, Wnext_dict, Us[iu])
                @show j Us[iu]
                push!(mpo_terms, (left_id, j) => k)
            end
            adjacency[:, iv] .= false
        end
        @assert !any(adjacency)

        @debug "terms" mpo_terms


        # @assert length(Wnext_dict) == length(W)
        elT = eltype(mpos)
        mpo_site = eltype(mpos)(undef, length(left_nodes), length(Wnext_dict))
        for ((i, j), k) in mpo_terms
            if SparseArraysBase.isstored(mpo_site, i, j)
                mpo_site[i, j] += convert(eltype(elT), k)
            else
                mpo_site[i, j] = k
            end
        end
        push!(mpos, mpo_site)
        @debug "transition at site $i" mpo_site

        @assert i == 1 || size(mpo_site, 1) == size(mpos[end - 1], 2)


        left_nodes = collect(keys(Wnext_dict))
    end

    return mpos
end

# Convenience overload accepting a GlobalOp directly.
function mpo_bond_optimizations(vertices::AbstractVector{Int}, ex::GlobalOp)
    return mpo_bond_optimizations(vertices, ttno_terms(vertices, ex))
end

# -----------------------------------------------------------------------
# Internal helper: build a SparseMatrixDOK from a (l_uid, r_uid) → Op dict.
# -----------------------------------------------------------------------
function _build_sparse_mpo(
        entries::Dict{Tuple{Int, Int}, LocalOp{T, Op}},
        left_uids::Vector{Int},
        right_uids::Vector{Int},
    ) where {T, Op}
    l_to_row = Dict(u => i for (i, u) in enumerate(left_uids))
    r_to_col = Dict(u => i for (i, u) in enumerate(right_uids))
    nrows = length(left_uids)
    ncols = length(right_uids)
    W = SparseArrayDOK{LocalOp{T, Op}}(undef, (nrows, ncols))
    for ((l, r), v) in entries
        W[l_to_row[l], r_to_col[r]] = v
    end
    return W
end
