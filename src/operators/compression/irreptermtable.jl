# The normal form (`canonicalize!`) and `ITOTermTable`, the `K × M` matrices the sweeps index into.

using TensorKit: Sector, unit

"""
    canonicalize!(ts::Terms) -> ts

Put `ts` in normal form in place: sort, sum coincident terms, drop cancelled ones. Idempotent.
"""
function canonicalize!(ts::Terms)
    isempty(ts.pending) || throw(
        ArgumentError(
            "this operator holds $(length(ts.pending)) unresolved on-site product(s) (`a[i] * b[i]`, " *
                "or a pair placed on one site), which are only resolved in the sites' bases when it " *
                "enters an OperatorSum: `opsum(lat, h)`. Until then it has no normal form, so its " *
                "length, terms and `≈` are not defined."
        )
    )
    _canonicalize!(ts.terms)
    return ts
end

# sort, merge adjacent equals, drop cancellations
function _canonicalize!(terms::Vector{Term{I}}) where {I}
    if length(terms) <= 1
        length(terms) == 1 && iszero(only(terms).coeff) && empty!(terms)   # so `isempty` cannot lie
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

    any(t -> iszero(t.coeff), terms) && filter!(t -> !iszero(t.coeff), terms)
    return terms
end

# Both in normal form, so a positional walk is a set comparison.
function _termsequal(ta::Vector{Term{I}}, tb::Vector{Term{I}}) where {I}
    return length(ta) == length(tb) && all(((x, y),) -> x == y && x.coeff == y.coeff, zip(ta, tb))
end

# `isapprox` of the two operators as sparse vectors over terms.
function _termsapprox(
        ta::Vector{Term{I}}, tb::Vector{Term{I}}; atol::Real = 0, rtol::Real = Base.rtoldefault(Float64)
    ) where {I}
    diff = Dict{Term{I}, ComplexF64}(t => t.coeff for t in ta)
    for t in tb
        diff[t] = get(diff, t, zero(ComplexF64)) - t.coeff
    end
    nrm(ts) = sqrt(sum(t -> abs2(t.coeff), ts; init = 0.0))
    return sqrt(sum(abs2, values(diff); init = 0.0)) <= max(atol, rtol * max(nrm(ta), nrm(tb)))
end

# Flat storage of a canonical operator: each term's active `(site, ITOKey)` factors in `K×M` matrices
# (`sites` zero-padded, ascending) plus `coeffs`. The caterpillar's bond charges are its fusion tree.
struct ITOTermTable{I <: Sector}
    sites::Matrix{Int}
    keys::Matrix{ITOKey{I}}
    coeffs::Vector{ComplexF64}
    nvertices::Int
end

arity(tt::ITOTermTable) = size(tt.sites, 1)
nterms(tt::ITOTermTable) = length(tt.coeffs)
nvertices(tt::ITOTermTable) = tt.nvertices

function ITOTermTable(ts::Terms{I}, N::Integer) where {I}
    canonicalize!(ts)
    terms = ts.terms
    M = length(terms)
    K = max(1, maximum(arity, terms; init = 0))   # the sweeps index row 1 unconditionally
    sitemat = zeros(Int, K, M)
    keymat = fill(ITOKey{I}(passthrough(I), unit(I), 1), K, M)
    for (t, term) in enumerate(terms)
        for j in 1:arity(term)
            sitemat[j, t] = term.sites[j]
            keymat[j, t] = term.keys[j]
        end
    end
    return ITOTermTable{I}(sitemat, keymat, ComplexF64[t.coeff for t in terms], Int(N))
end

# `ITOKey` at site `s` of term `t`: the stored key on an active site, else the pass-through carrying
# the running bond charge to its left (which preserves block-diagonality).
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
