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
using TensorKit: Vect, ElementarySpace, fusiontrees, permute, dim, unit, isomorphism, @tensor
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

# NOTE: this sweep deliberately mirrors `mpo_bond_optimizations(..., ::BipartiteAlgorithm)` in
# `statemachines/graphbuilding.jl`; it is kept separate (rather than sharing code) so the
# dense/Pauli path stays byte-for-byte untouched. The only ITO-specific additions are: emitting the
# `ITOKey.op` letter (not the whole key) and tracking each retained bond's charge (`secW`). Keep the
# two in sync if the bipartite sweep changes.
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
the active ITO letters (skipping pass-through), and rebuild each term's caterpillar tree from the
per-active-site outgoing bond charges (`_tree_from_bonds`; multiplicity-free ⇒ trivial vertices).
For the lossless bipartite compression this recovers the original `TermSum` exactly — the
faithfulness check.
"""
function mpo_terms(
        Ws::Vector{<:SparseMatrixDOK{LocalOp{ComplexF64, IrrepOperator{I}}}},
        bondsectors::Vector{Vector{I}}
    ) where {I}
    N = length(Ws)
    d = Dictionary{TermKey{I, Int}, ComplexF64}()
    N == 0 && return TermSum{I, Int, ComplexF64}(d)

    function walk(i, leftidx, coeff, letters, bonds)
        if i > N
            sites = Int[]
            ops = IrrepOperator{I}[]
            for (k, op) in enumerate(letters)
                ispassthrough(op) || (push!(sites, k); push!(ops, op))
            end
            tree = _tree_from_bonds([o.c for o in ops], bonds, ones(Int, length(ops)))
            setwith!(+, d, TermKey{I, Int}(sites, ops, tree), coeff)
            return
        end
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            l == leftidx || continue
            for (letter, c) in _local_terms(localop)
                letter === nothing && continue
                # record the outgoing bond charge for active (non-pass-through) letters — these are
                # the caterpillar running bonds `[b₁ … b_K]`
                newbonds = ispassthrough(letter) ? bonds : push!(copy(bonds), bondsectors[i][r])
                walk(i + 1, r, coeff * c, push!(copy(letters), letter), newbonds)
            end
        end
        return
    end
    walk(1, 1, ComplexF64(1), IrrepOperator{I}[], I[])
    return TermSum{I, Int, ComplexF64}(d)
end

# Phase 5: assemble symmetric MPO TensorMaps from the reduced bond data
# =====================================================================
# Each site tensor is `W_i : V_i ⊗ B_i ← B_{i-1} ⊗ V_i`, a symmetric `TensorMap` whose virtual
# legs `B` are `GradedSpace`s built from the per-sector bond multiplicities. Contracting the chain
# (trivial boundary bonds) recovers the operator.

# GradedSpace on a bond with the given per-index charges (multiplicity = count per sector).
function _bond_space(sec::Vector{I}) where {I}
    counts = Dict{I, Int}()
    for c in sec
        counts[c] = get(counts, c, 0) + 1
    end
    return Vect[I](counts)
end

# degeneracy index (within its charge sector) of each bond index
function _deg_indices(sec::Vector{I}) where {I}
    counts = Dict{I, Int}()
    out = Vector{Int}(undef, length(sec))
    for (j, c) in enumerate(sec)
        counts[c] = get(counts, c, 0) + 1
        out[j] = counts[c]
    end
    return out
end

"""
    irrep_mpo_tensors(Ws, bondsectors, sites) -> Vector{<:AbstractTensorMap}

Assemble the symmetric MPO from the reduced bond matrices + bond sectors (from `irrep_mpo`). Site
tensor `W_i : V_i ⊗ B_{i-1} ← B_i ⊗ V_i`; the boundary bonds `B_0`, `B_N` are one-dimensional.
Each reduced entry `(l, r, letter, coeff)` places the ITO letter's tensor into the `(l → r)` bond
transition weighted by `coeff`, coupling the operator charge into the bond via the (forward)
fusion `b_L ⊗ c → b_R`.
"""
function irrep_mpo_tensors(
        Ws::Vector{<:SparseMatrixDOK{LocalOp{ComplexF64, IrrepOperator{I}}}},
        bondsectors::Vector{Vector{I}}, sites
    ) where {I}
    N = length(Ws)
    # `map` infers a concrete `Vector{<:AbstractTensorMap}` element type (all site tensors share
    # the same `V ⊗ B ← B ⊗ V` space *type*), unlike an untyped preallocated buffer.
    return map(1:N) do i
        V = sites[i]
        secL = i == 1 ? I[unit(I)] : bondsectors[i - 1]
        secR = bondsectors[i]
        Bleft = _bond_space(secL)
        Bright = _bond_space(secR)
        degL = i == 1 ? Int[1] : _deg_indices(bondsectors[i - 1])
        degR = _deg_indices(secR)

        W = zeros(ComplexF64, V ⊗ Bleft ← Bright ⊗ V)
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            bL, dL = secL[l], degL[l]
            bR, dR = secR[r], degR[r]
            for (letter, coeff) in _local_terms(localop)
                letter === nothing && continue
                c = letter.c
                Vc = Vect[I](c => 1)
                # V ← V ⊗ Vect[c]; pass-through (c = unit) carries a trivial charge leg so the
                # bond contraction is uniform (instantiate would drop it, returning bare id(V)).
                O = ispassthrough(letter) ? isomorphism(ComplexF64, V ← V ⊗ Vc) :
                    instantiate(letter, V)
                # bond coupler κ : Bleft ⊗ Vc ← Bright, forward fusion (bL, c) → bR — running bond
                # FIRST, then the operator charge, matching the left-nested caterpillar the oracle
                # uses. (Charge-first would agree only for symmetric vertices, e.g. K=2 singlets, and
                # flips the sign at antisymmetric inner vertices like 1⊗1→1 for K ≥ 3.)
                κ = zeros(ComplexF64, Bleft ⊗ Vc ← Bright)
                f1 = only(fusiontrees((bL, c), bR, (false, false)))
                f2 = only(fusiontrees((bR,), bR, (false,)))
                κ[f1, f2][dL, 1, dR] = coeff
                @tensor Wentry[o bl; br ii] := O[o; ii cc] * κ[bl cc; br]
                W = W + Wentry
            end
        end
        return W
    end
end
