# Interning of the term table's contiguous column suffixes/prefixes, shared by every sweep backend.

# Walk/insert-on-miss behind `_suffix_ids`, `_prefix_ids` and `_rel_suffix_ids`. `dir = -1` walks
# columns `K:-1:1`, writing `id[j, t]` from the already-interned tail `id[j + 1, t]` (padding only
# trails, so a zero column is skipped); `dir = +1` walks `1:K`, writing `id[j + 1, t]` and stopping at
# the first zero. `keyfn(j, t, s)` builds the tuple to intern from column `j` (site `s`).
function _intern_columns!(intern::Dictionary, id::Matrix{Int}, tt::ITOTermTable, M::Int, K::Int, dir::Int, keyfn)
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

"""
    _suffix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Intern every term's *contiguous column suffixes*: `sufid[j, t]` is a dense id for the factor list
`j:K` of term `t` (`0` for the exhausted suffix), built bottom-up in `Θ(M·K)`.

Equal ids mean equal remaining factor lists, which is *not* equal suffix paths: `_op_at_ito` fills the
idle sites before the first remaining factor with a pass-through carrying the running bond charge. A
suffix path is therefore identified by `(sufid[j₀, t], running bond charge)`.
"""
function _suffix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    sufid = zeros(Int, K + 1, M)      # row K+1 and every padded position stay 0 == exhausted
    intern = Dictionary{Tuple{Int, ITOKey{I}, Int}, Int}()
    _intern_columns!(intern, sufid, tt, M, K, -1, (j, t, s) -> (s, tt.keys[j, t], sufid[j + 1, t]))
    return sufid
end

"""
    _prefix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Mirror image of [`_suffix_ids`](@ref): `preid[j, t]` is a dense id for the factor list `1:j-1` of
term `t` (`preid[1, t] == 0`, the empty prefix). Equal ids *are* equal prefix paths, with no charge
component needed: the pass-through at every idle site of a prefix carries a running charge fixed by
the factors to its left. Used by the `IndependentSVD` sweep.
"""
function _prefix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    preid = zeros(Int, K + 1, M)      # row 1 stays 0 == the empty prefix
    intern = Dictionary{Tuple{Int, Int, ITOKey{I}}, Int}()
    _intern_columns!(intern, preid, tt, M, K, 1, (j, t, s) -> (preid[j, t], s, tt.keys[j, t]))
    return preid
end

"""
    _rel_suffix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Translation-invariant twin of [`_suffix_ids`](@ref): interns suffixes by *shape*, consing
`(gap to the next active site, ITOKey, tail id)` instead of the absolute site. Comparing suffix
classes across bonds needs the positional information split off; `(distance, relsufid, running
charge)` is the canonical name of a right vertex, see [`_rdesc`](@ref).
"""
function _rel_suffix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    relid = zeros(Int, K + 1, M)
    intern = Dictionary{Tuple{Int, ITOKey{I}, Int}, Int}()
    _intern_columns!(
        intern, relid, tt, M, K, -1,
        function (j, t, s)
            nxt = j < K ? tt.sites[j + 1, t] : 0
            return (iszero(nxt) ? 0 : nxt - s, tt.keys[j, t], relid[j + 1, t])   # gap 0 == last factor
        end
    )
    return relid
end
