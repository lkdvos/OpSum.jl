# The normal form (`canonicalize!`) and `ITOTermTable`, the `K × M` matrices the sweeps index into.

using TensorKit: Sector, unit

"""
    canonicalize!(H::TermSum) -> H

Put `H` in normal form **in place**: sort its terms (by sites, then keys, so coincident ones become
adjacent), sum coincident ones, drop cancelled ones.

Assumes nothing about its input, so no flag can fall out of step with the terms. It is idempotent, but
a repeat call re-sorts rather than returning early — hence the sweep normalises once, at the
[`ITOTermTable`](@ref) boundary.
"""
canonicalize!(H::TermSum) = (_canonicalize!(H.terms); H)

"""
    canonicalize!(ts::Terms) -> ts

The same normal form on a bag of terms. Bags are not normalised as they are built, so this is what
`≈` and `==` on them go through.
"""
canonicalize!(ts::Terms) = (_canonicalize!(ts.terms); ts)

# sort, merge adjacent equals, drop cancellations
function _canonicalize!(terms::Vector{Term{I}}) where {I}
    if length(terms) <= 1
        # still drop a lone cancelled term, so `isempty` cannot lie
        length(terms) == 1 && iszero(only(terms).coeff) && empty!(terms)
        return terms
    end

    sort!(terms)
    out = 0
    @inbounds for i in eachindex(terms)
        t = terms[i]
        if out >= 1 && terms[out] == t
            prev = terms[out]
            terms[out] = Term{I}(prev.sites, prev.keys, prev.coeff + t.coeff)
        else
            out += 1
            terms[out] = t
        end
    end
    resize!(terms, out)

    # cancellations are rarer than duplicates, so only pay for a second pass when there are any
    any(t -> iszero(t.coeff), terms) && filter!(t -> !iszero(t.coeff), terms)
    return terms
end

# Already in normal form and sorted the same way, so a positional walk is a set comparison.
function _termsapprox(ta::Vector{Term{I}}, tb::Vector{Term{I}}; kwargs...) where {I}
    length(ta) == length(tb) || return false
    for (x, y) in zip(ta, tb)
        x == y || return false
        isapprox(x.coeff, y.coeff; kwargs...) || return false
    end
    return true
end

function _termsequal(ta::Vector{Term{I}}, tb::Vector{Term{I}}) where {I}
    length(ta) == length(tb) || return false
    return all(((x, y),) -> x == y && x.coeff == y.coeff, zip(ta, tb))
end

# the symbol an inactive (padded) slot carries, matching `_op_at_ito`'s reconstruction
_padkey(::Type{I}) where {I <: Sector} = ITOKey{I}(passthrough(I), unit(I), 1)

"""
    ITOTermTable{I<:Sector}

Flat, sparse-per-term storage of a canonical ITO operator on an `N`-vertex chain: each term's active
`(site, ITOKey)` factors in `K×M` matrices (`sites` zero-padded, ascending) plus a parallel `coeffs`
vector. This is the only thing the reduced-MPO sweeps read.

Note there is no fusion tree here, and none is needed: an `ITOKey` carries the running bond charge
and vertex label at its position, and for the left-nested (caterpillar) coupling this algebra
supports those *are* the tree — `bondcharges`/`vertexlabels` read them off it, `_tree_from_bonds`
puts it back together. irrepkey.jl has the argument, including why the total charge alone would not
do.
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

"""
    ITOTermTable(H::TermSum)

Materialise `H` in normal form ([`canonicalize!`](@ref)) as the flat table the MPO sweep consumes, on
the `N = length(lattice(H))` sites it is defined over.
"""
function ITOTermTable(H::TermSum{I}) where {I}
    canonicalize!(H)
    terms = H.terms
    N = length(lattice(H))
    M = length(terms)
    # `arity(tt) ≥ 1`: the sweeps index row 1 unconditionally.
    K = max(1, maximum(arity, terms; init = 0))
    sitemat = zeros(Int, K, M)
    keymat = fill(_padkey(I), K, M)
    for (t, term) in enumerate(terms)
        for j in 1:arity(term)
            sitemat[j, t] = term.sites[j]
            keymat[j, t] = term.keys[j]
        end
    end
    return ITOTermTable{I}(sitemat, keymat, ComplexF64[t.coeff for t in terms], N)
end

# `ITOKey` at site `s` of term `t`: the stored key on an active site, else the pass-through symbol
# carrying the running bond charge to its left — which is what preserves block-diagonality.
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
