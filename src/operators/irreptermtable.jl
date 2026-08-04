# Flat term-list storage for the ITO automaton
# ============================================
# `ITOTermTable` stores each term's active `(site, ITOKey)` factors in flat matrices (the ITO
# counterpart of the dense `TermTable`). The reduced-MPO sweeps that consume it live in
# irrepgraph.jl; `irrep_mpo` (irrepmpo.jl) is the public entry.
#
# The one ITO-specific subtlety vs the dense `TermTable`: idle sites are NOT a constant identity —
# they carry a pass-through symbol with the *running* bond charge. The flat store keeps only active
# factors; `_op_at_ito` reconstructs the pass-through key at any idle position from the last active
# bond charge to its left, so the per-bond transition key is effectively `(prev_id, op, bond,
# vertex)`, preserving block-diagonality exactly.

using TensorKit: Sector, unit

"""
    ITOTermTable{I<:Sector}

Flat, sparse-per-term storage of an ITO term-sum on an `N`-vertex chain: each term's active
`(site, ITOKey)` factors in `K×M` matrices (`sites` zero-padded, ascending) plus a parallel
`coeffs` vector.
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
