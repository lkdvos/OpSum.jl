# `TermTable`, the `K × M` matrices the sweeps index into, and the interning of their columns.

using TensorKit: Sector, unit

# Flat storage of a canonical operator: each term's active `(site, Factor)` factors in `K×M` matrices
# (`sites` zero-padded, ascending) plus `coeffs`. The caterpillar's bond charges are its fusion tree.
struct TermTable{I <: Sector}
    sites::Matrix{Int}
    factors::Matrix{Factor{I}}
    coeffs::Vector{ComplexF64}
    nvertices::Int
end

arity(tt::TermTable) = size(tt.sites, 1)
nterms(tt::TermTable) = length(tt.coeffs)
nvertices(tt::TermTable) = tt.nvertices

function TermTable(ts::Terms{I}, N::Integer) where {I}
    canonicalize!(ts)
    terms = ts.terms
    M = length(terms)
    K = max(1, maximum(arity, terms; init = 0))   # the sweeps index row 1 unconditionally
    sitemat = zeros(Int, K, M)
    keymat = fill(Factor{I}(passthrough(I), unit(I)), K, M)
    for (t, term) in enumerate(terms)
        for j in 1:arity(term)
            sitemat[j, t] = term.sites[j]
            keymat[j, t] = term.factors[j]
        end
    end
    return TermTable{I}(sitemat, keymat, ComplexF64[t.coeff for t in terms], Int(N))
end

# `Factor` at site `s` of term `t`: the stored key on an active site, else the pass-through carrying
# the running bond charge to its left (which preserves block-diagonality).
function _factor_at(tt::TermTable{I}, t::Int, s::Int) where {I}
    lastbond = unit(I)
    @inbounds for j in 1:size(tt.sites, 1)
        st = tt.sites[j, t]
        st == 0 && break
        if st == s
            return tt.factors[j, t]
        elseif st < s
            lastbond = tt.factors[j, t].bond
        else
            break
        end
    end
    return Factor{I}(passthrough(I), lastbond)
end

# Interning of the term table's contiguous column suffixes/prefixes, shared by every sweep backend.

# Walk/insert-on-miss behind the three id matrices. `dir = -1` walks columns `K:-1:1` writing
# `id[j, t]`; `dir = +1` walks `1:K` writing `id[j + 1, t]` and stopping at padding. `keyfn(j, t, s)`
# builds the tuple to intern from column `j` (site `s`).
function _intern_columns!(intern::Dictionary, id::Matrix{Int}, tt::TermTable, M::Int, K::Int, dir::Int, keyfn)
    nid = 0
    for t in 1:M
        for j in (dir < 0 ? (K:-1:1) : (1:K))
            s = tt.sites[j, t]
            iszero(s) && (dir < 0 ? continue : break)
            trans = keyfn(j, t, s)
            v = get(intern, trans, 0)
            if iszero(v)
                v = (nid += 1)
                insert!(intern, trans, v)
            end
            id[dir < 0 ? j : j + 1, t] = v
        end
    end
    return id
end

# `sufid[j, t]`: id of the factor list `j:K` of term `t` (0 = exhausted). Equal ids mean equal factor
# lists, not equal suffix paths: idle sites carry the running bond charge, so a path is
# `(sufid[j, t], running bond charge)`.
function _suffix_ids(tt::TermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    sufid = zeros(Int, K + 1, M)      # row K+1 and every padded position stay 0 == exhausted
    intern = Dictionary{Tuple{Int, Factor{I}, Int}, Int}()
    _intern_columns!(intern, sufid, tt, M, K, -1, (j, t, s) -> (s, tt.factors[j, t], sufid[j + 1, t]))
    return sufid
end

# `preid[j, t]`: id of the factor list `1:j-1` (0 = empty). Equal ids are equal prefix paths (the
# running charge is fixed by the factors). Used by `IndependentSVD`.
function _prefix_ids(tt::TermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    preid = zeros(Int, K + 1, M)      # row 1 stays 0 == the empty prefix
    intern = Dictionary{Tuple{Int, Int, Factor{I}}, Int}()
    _intern_columns!(intern, preid, tt, M, K, 1, (j, t, s) -> (preid[j, t], s, tt.factors[j, t]))
    return preid
end

# Translation-invariant `_suffix_ids`: conses `(gap to the next factor, Factor, tail id)` instead of
# the absolute site; with distance and charge it names a right vertex (`_rdesc`).
function _rel_suffix_ids(tt::TermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    relid = zeros(Int, K + 1, M)
    intern = Dictionary{Tuple{Int, Factor{I}, Int}, Int}()
    _intern_columns!(
        intern, relid, tt, M, K, -1,
        function (j, t, s)
            nxt = j < K ? tt.sites[j + 1, t] : 0
            return (iszero(nxt) ? 0 : nxt - s, tt.factors[j, t], relid[j + 1, t])   # gap 0 == last factor
        end
    )
    return relid
end
