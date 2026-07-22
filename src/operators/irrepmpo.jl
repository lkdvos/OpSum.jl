# Phase 4: per-bond-sector bond optimization for the ITO automaton
# ================================================================
# Compress the charge-augmented trie (from `irrep_trie`) into a reduced MPO via the same
# bipartite min-vertex-cover sweep as the dense pipeline, with two differences:
#
# * emitted entries carry the ITO *letter* `ITOKey.op` (an `IrrepOperator`), not the whole key,
#   so a site tensor is a `SparseMatrixDOK{LocalOp{ComplexF64, IrrepOperator{I}}}` of *reduced*
#   coefficients — the input Phase 5 will wrap into symmetric `TensorMap`s;
# * each retained bond index is labelled by its bond charge (`ITOKey.bond`). Because the trie key
#   already carries the bond charge, the automaton is block-diagonal in it: every bond index has a
#   single, well-defined charge. That invariant is `@assert`ed (not enforced by re-partitioning),
#   which also yields the per-sector bond dimensions directly.
#
# Supported scope follows Phase 2/3 (K ≤ 2 active sites per term). The dense/Pauli
# `mpo_bond_optimizations` pipeline is untouched.

using SparseArraysBase: SparseMatrixDOK, storedpairs
using .IrrepTensorOperators: IrrepOperator

"""
    irrep_mpo(H::TermSum, sites) -> (Ws, bondsectors)

Compress the ITO Hamiltonian `H` over `N = length(sites)` lattice sites into a reduced MPO.

Returns `Ws::Vector{SparseMatrixDOK{LocalOp{ComplexF64, IrrepOperator{I}}}}` (one reduced bond
matrix per site; entries are ITO letters times reduced coefficients) and `bondsectors::Vector{
Vector{I}}` where `bondsectors[i]` gives the bond charge of each bond index to the right of site
`i` (so `size(Ws[i], 2) == length(bondsectors[i])`). The left boundary (bond 0) is a single
trivial-charge index.
"""
function irrep_mpo(H::TermSum{I, S, Tc}, sites) where {I, S, Tc}
    return _irrep_bipartite(irrep_trie(H, sites), length(sites))
end

function _irrep_bipartite(prefix_trie::Trie{ITOKey{I}, T}, N::Int) where {I, T}
    Op = ITOKey{I}
    LOp = LocalOp{ComplexF64, IrrepOperator{I}}
    isempty(prefix_trie) && return (SparseMatrixDOK{LOp}[], Vector{I}[])

    left_nodes = Trie{Op, T}[prefix_trie]
    sizes = Tuple{Int, Int}[]
    dicts = Dictionary{CartesianIndex{2}, LOp}[]
    bondsectors = Vector{I}[]

    for i in 1:N
        Us = Trie{Op, T}[]
        parents = Pair{Trie{Op, T}, Op}[]
        uidx_ = Int[]
        uidx__ = 0
        mpo_terms = Pair{CartesianIndex{2}, LOp}[]
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
                iv = get!(uid!, Vs, Op[])
                push!(nonzero_list, CartesianIndex(iu, iv) => something(U.value))
            else
                for (operator, coeff) in pairs(U)
                    iv = get!(uid!, Vs, operator)
                    push!(nonzero_list, CartesianIndex(iu, iv) => coeff)
                end
            end
        end

        coefficients = zeros(T, length(Us), uid!.current)
        for (Idx, c) in nonzero_list
            @assert iszero(coefficients[Tuple(Idx)...])
            coefficients[Tuple(Idx)...] = c
        end
        adjacency = (!iszero).(coefficients)
        coverU, coverV, _ = min_vertex_cover_bipartite(adjacency)

        # bond charge out of site i for each U (= its parent transition's bond charge)
        ubond = I[parents[iu][2].bond for iu in 1:length(Us)]

        W = Us[coverU]
        adjacency[coverU, :] .= false
        uidnext! = Counter()
        Wnext_dict = IdDict{Trie{Op, T}, Int}()
        secW = I[]

        for iu in findall(coverU)
            left_id = uidx_[iu]
            (_, k) = parents[iu]
            j = get!(uidnext!, Wnext_dict, Us[iu])
            j > length(secW) && push!(secW, ubond[iu])
            if i == N
                c = coefficients[iu, 1]
                push!(mpo_terms, CartesianIndex(left_id, j) => k.op * c)
            else
                push!(mpo_terms, CartesianIndex(left_id, j) => convert(LOp, k.op))
            end
        end

        Vkeys = collect(keys(Vs))
        for iv in findall(coverV)
            suffix = Vkeys[iv]
            newnode = typeof(prefix_trie)()
            push!(W, newnode)
            node = newnode
            for s in suffix
                node = _add_child!(node, s)
            end
            node.value = one(T)

            # charge of this bond index = common charge of all terms routed through it
            conn = findall(!iszero, coefficients[:, iv])
            charge = ubond[first(conn)]
            @assert all(ubond[iu] == charge for iu in conn) "bond index not sector-pure (block-diagonality violated)"
            push!(secW, charge)

            j = uidnext!()
            for iu in findall(adjacency[:, iv])
                left_id = uidx_[iu]
                (_, k) = parents[iu]
                c = coefficients[iu, iv]
                push!(mpo_terms, CartesianIndex(left_id, j) => k.op * c)
            end
            adjacency[:, iv] .= false
        end
        @assert !any(adjacency)
        @assert length(secW) == length(W)

        site_dict = Dictionary{CartesianIndex{2}, LOp}()
        for (ij, k) in mpo_terms
            increaseindex!(site_dict, ij, k)
        end
        push!(sizes, (length(left_nodes), length(W)))
        push!(dicts, site_dict)
        push!(bondsectors, secW)

        @assert i == 1 || sizes[end][1] == sizes[end - 1][2]

        left_nodes = W
    end

    return (map(SparseArraysBase.sparse, dicts, sizes), bondsectors)
end

"""
    mpo_terms(Ws, bondsectors) -> TermSum{I, Int, ComplexF64}

Reconstruct the term-sum generated by a reduced MPO (inverse of `irrep_mpo` at the reduced level):
enumerate every path through the bond matrices, multiply the reduced coefficients along it, read
the active ITO letters (skipping pass-through) and the total charge from the boundary sector. For
the lossless bipartite compression this recovers the original `TermSum` exactly — the faithfulness
check.
"""
function mpo_terms(
        Ws::Vector{<:SparseMatrixDOK{LocalOp{ComplexF64, IrrepOperator{I}}}},
        bondsectors::Vector{Vector{I}}
    ) where {I}
    N = length(Ws)
    d = Dictionary{TermKey{I, Int}, ComplexF64}()
    N == 0 && return TermSum{I, Int, ComplexF64}(d)
    total = only(bondsectors[N])

    function walk(i, leftidx, coeff, letters)
        if i > N
            sites = Int[]
            ops = IrrepOperator{I}[]
            for (k, op) in enumerate(letters)
                ispassthrough(op) || (push!(sites, k); push!(ops, op))
            end
            setwith!(+, d, TermKey{I, Int}(sites, ops, total), coeff)
            return
        end
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            l == leftidx || continue
            for (letter, c) in _local_terms(localop)
                letter === nothing && continue
                walk(i + 1, r, coeff * c, push!(copy(letters), letter))
            end
        end
        return
    end
    walk(1, 1, ComplexF64(1), IrrepOperator{I}[])
    return TermSum{I, Int, ComplexF64}(d)
end
