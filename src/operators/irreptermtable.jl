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

using TensorKit: Sector, unit

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
        for (Idx, c) in nonzero_list
            @assert iszero(coefficients[Tuple(Idx)...])
            coefficients[Tuple(Idx)...] = c
        end
        adjacency = (!iszero).(coefficients)
        coverU, coverV, _ = min_vertex_cover_bipartite(adjacency)

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
