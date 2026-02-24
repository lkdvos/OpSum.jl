struct NormalLeftNode
    uid::Int
end

struct ComplementaryLeftNode
    uid::Int
end

const LeftNode = Union{NormalLeftNode, ComplementaryLeftNode}

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

    # -----------------------------------------------------------------------
    # Step 1: Build suffix trie
    # suffix_uids[o, i] = UID of ops[o][(i+1):N]
    # -----------------------------------------------------------------------
    uid_counter = Ref(0)
    next_uid!() = (uid_counter[] += 1; uid_counter[])

    # -----------------------------------------------------------------------
    # Step 1: Build suffix trie (reversed suffixes for incremental traversal)
    # suffix_uids[o, i] = UID of ops[o][(i+1):N]
    #
    # The suffix at bond i is obtained from bond i+1 by prepending ops[o][i+1].
    # Storing reversed suffixes as trie paths lets each step simply follow (or
    # create) one child edge, dropping allocation from O(N-i) to O(1) per step.
    # -----------------------------------------------------------------------
    suffix_trie = Trie{Op, Int}()
    suffix_uids = zeros(Int, K, N)

    # Empty suffix (i=N): assign a single shared UID at the trie root.
    suffix_trie.value = next_uid!()
    fill!(@view(suffix_uids[:, N]), suffix_trie.value)

    # Per-term pointer; all start at root and advance one edge per bond.
    suffix_nodes = fill(suffix_trie, K)
    for i in (N - 1):-1:1
        for o in 1:K
            node = get!(() -> Trie{Op, Int}(), suffix_nodes[o].children, terms[o].ops[i + 1])
            if isnothing(node.value)
                node.value = next_uid!()
            end
            suffix_nodes[o] = node
            suffix_uids[o, i] = node.value
        end
    end

    # -----------------------------------------------------------------------
    # Step 2: Build prefix trie (standard left-to-right, also incremental)
    # prefix_uids[o, i] = UID of ops[o][1:i]
    #
    # The prefix at bond i is obtained from bond i-1 by appending ops[o][i],
    # which is the natural trie direction — no reversal needed.
    # -----------------------------------------------------------------------
    prefix_trie = Trie{Op, Int}()
    prefix_uids = zeros(Int, K, N)

    prefix_nodes = fill(prefix_trie, K)
    for i in 1:N
        for o in 1:K
            node = get!(() -> Trie{Op, Int}(), prefix_nodes[o].children, terms[o].ops[i])
            if isnothing(node.value)
                node.value = next_uid!()
            end
            prefix_nodes[o] = node
            prefix_uids[o, i] = node.value
        end
    end

    @debug "debugging preprocessing" prefix_uids suffix_uids terms

    # -----------------------------------------------------------------------
    # Step 3: Initialise sweep state
    # -----------------------------------------------------------------------
    EMPTY_UID = next_uid!()
    term_carrier = fill(EMPTY_UID, K)
    active_left = Dict{Int, LeftNode}(EMPTY_UID => NormalLeftNode(EMPTY_UID))
    Ws = SparseMatrixDOK{LocalOp{T, Op}}[]

    # -----------------------------------------------------------------------
    # Step 4: Main sweep — bonds i = 1 .. N-1
    # At bond i we build W[i], mapping bond-(i-1) states → bond-i states.
    # -----------------------------------------------------------------------
    for i in 1:(N - 1)
        left_uids = sort(unique(term_carrier))
        right_uids = sort(unique(@view(suffix_uids[:, i])))

        l_idx = Dict(u => j for (j, u) in enumerate(left_uids))
        r_idx = Dict(v => j for (j, v) in enumerate(right_uids))

        # 4a: binary adjacency matrix for bipartite matching
        adjacency = falses(length(left_uids), length(right_uids))
        for o in 1:K
            adjacency[l_idx[term_carrier[o]], r_idx[suffix_uids[o, i]]] = true
        end
        @debug "adjacency at site $i" adjacency left_uids right_uids term_carrier

        # 4b: minimum vertex cover
        coverU, coverV, _, _, _ = min_vertex_cover_bipartite(adjacency)
        @debug "covers" coverU coverV

        # 4c: assign new carriers and build W[i] entries
        W_entries = Dict{Tuple{Int, Int}, LocalOp{T, Op}}()
        new_active = Dict{Int, LeftNode}()
        term_new_carrier = zeros(Int, K)
        right_to_comp = Dict{Int, Int}()   # r_uid → complementary uid

        for o in 1:K
            l = term_carrier[o]
            iu = l_idx[l]
            op_i = LocalOp{T, Op}(terms[o].ops[i])

            if coverU[iu]
                # Normal pass-through: carrier advances to next prefix UID.
                # All terms with the same (l, nc) share identical ops[i]
                # (same prefix ⟹ same local op), so set once.
                nc = prefix_uids[o, i]
                term_new_carrier[o] = nc
                if !haskey(new_active, nc)
                    new_active[nc] = NormalLeftNode(nc)
                end
                key = (l, nc)
                if !haskey(W_entries, key)
                    W_entries[key] = op_i
                end
            else
                # Complementary transition.
                # If l is a NormalLeftNode, the coefficient is placed here (first entry for this term).
                # If l is a ComplementaryLeftNode, the coefficient was already placed when l was
                # created; use bare ops and set-once to avoid double-counting merged terms.
                r = suffix_uids[o, i]
                if !haskey(right_to_comp, r)
                    c = next_uid!()
                    right_to_comp[r] = c
                    new_active[c] = ComplementaryLeftNode(c)
                end
                c = right_to_comp[r]
                term_new_carrier[o] = c
                key = (l, c)
                if active_left[l] isa ComplementaryLeftNode
                    # Coefficient already absorbed when l was created; bare ops, set-once.
                    if !haskey(W_entries, key)
                        W_entries[key] = op_i
                    end
                else
                    # Normal l: place coefficient here, accumulate (handles different ops[i]).
                    scaled_op = terms[o].coeff * op_i
                    if haskey(W_entries, key)
                        W_entries[key] = W_entries[key] + scaled_op
                    else
                        W_entries[key] = scaled_op
                    end
                end
            end
        end

        # 4d: update carriers
        term_carrier .= term_new_carrier

        # 4e: store W[i]
        active_left = new_active
        right_uids_new = sort(collect(keys(new_active)))
        push!(Ws, _build_sparse_mpo(W_entries, left_uids, right_uids_new))
        @debug "W at site $(length(Ws))" last(Ws)
    end

    # -----------------------------------------------------------------------
    # Step 5: Final site W[N]
    # Coefficient is placed here for any term still in a normal (prefix) state.
    # -----------------------------------------------------------------------
    final_left_uids = sort(collect(keys(active_left)))
    W_final = Dict{Tuple{Int, Int}, LocalOp{T, Op}}()
    for o in 1:K
        l = term_carrier[o]
        key = (l, 1)
        op_N = LocalOp{T, Op}(terms[o].ops[N])
        if active_left[l] isa ComplementaryLeftNode
            # Coefficient was placed when l was created; bare ops, set-once.
            if !haskey(W_final, key)
                W_final[key] = op_N
            end
        else
            # Normal carrier: place coefficient here; accumulate (terms may share prefix
            # but differ in ops[N], producing a Sum LocalOp).
            scaled_op = terms[o].coeff * op_N
            if haskey(W_final, key)
                W_final[key] = W_final[key] + scaled_op
            else
                W_final[key] = scaled_op
            end
        end
    end
    push!(Ws, _build_sparse_mpo(W_final, final_left_uids, [1]))

    return Ws
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
