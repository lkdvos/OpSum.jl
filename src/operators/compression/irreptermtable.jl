# The normal form (`canonicalize!`) and `ITOTermTable`, the `K × M` matrices the sweeps index into.

using TensorKit: Sector, unit

"""
    canonicalize!(ts::Terms) -> ts

Put `ts` in normal form **in place**: sort its terms (by sites, then keys, so coincident ones become
adjacent), sum coincident ones, drop cancelled ones.

Assumes nothing about its input and is idempotent, but a repeat call re-sorts rather than returning
early, hence the sweep normalises once, at the [`ITOTermTable`](@ref) boundary. Everything that
observes the term set (`length`, iteration, `≈`, `==`, `show`) goes through this.
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

    # cancellations are rare: only pay for a second pass when there are any
    any(t -> iszero(t.coeff), terms) && filter!(t -> !iszero(t.coeff), terms)
    return terms
end

# Already in normal form and sorted the same way, so a positional walk is a set comparison.
function _termsequal(ta::Vector{Term{I}}, tb::Vector{Term{I}}) where {I}
    return length(ta) == length(tb) && all(((x, y),) -> x == y && x.coeff == y.coeff, zip(ta, tb))
end

# The two operators as sparse vectors over terms (a missing term has coefficient zero), compared like
# `isapprox` on vectors: `‖a - b‖ ≤ max(atol, rtol · max(‖a‖, ‖b‖))`.
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

"""
    ITOTermTable{I<:Sector}

Flat, sparse-per-term storage of a canonical ITO operator on an `N`-vertex chain: each term's active
`(site, ITOKey)` factors in `K×M` matrices (`sites` zero-padded, ascending) plus a parallel `coeffs`
vector. This is the only thing the reduced-MPO sweeps read.

There is no fusion tree here: an `ITOKey` carries the running bond charge and vertex label, which for
the caterpillar coupling *are* the tree (see irrepkey.jl).
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
    ITOTermTable(ts::Terms, N::Int)

Materialise `ts` in normal form ([`canonicalize!`](@ref)) as the flat table the MPO sweep consumes, on
`N` sites. `N` is the only thing the sweep needs from the lattice.
"""
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
