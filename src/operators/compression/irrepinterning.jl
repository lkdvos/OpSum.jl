# Interning of the term table's contiguous column suffixes/prefixes, shared by every sweep backend.

# Walk/insert-on-miss behind the three id matrices. `dir = -1` walks columns `K:-1:1` writing
# `id[j, t]`; `dir = +1` walks `1:K` writing `id[j + 1, t]` and stopping at padding. `keyfn(j, t, s)`
# builds the tuple to intern from column `j` (site `s`).
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

# `sufid[j, t]`: id of the factor list `j:K` of term `t` (0 = exhausted). Equal ids mean equal factor
# lists, not equal suffix paths: idle sites carry the running bond charge, so a path is
# `(sufid[j, t], running bond charge)`.
function _suffix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    sufid = zeros(Int, K + 1, M)      # row K+1 and every padded position stay 0 == exhausted
    intern = Dictionary{Tuple{Int, ITOKey{I}, Int}, Int}()
    _intern_columns!(intern, sufid, tt, M, K, -1, (j, t, s) -> (s, tt.keys[j, t], sufid[j + 1, t]))
    return sufid
end

# `preid[j, t]`: id of the factor list `1:j-1` (0 = empty). Equal ids are equal prefix paths (the
# running charge is fixed by the factors). Used by `IndependentSVD`.
function _prefix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    preid = zeros(Int, K + 1, M)      # row 1 stays 0 == the empty prefix
    intern = Dictionary{Tuple{Int, Int, ITOKey{I}}, Int}()
    _intern_columns!(intern, preid, tt, M, K, 1, (j, t, s) -> (preid[j, t], s, tt.keys[j, t]))
    return preid
end

# Translation-invariant `_suffix_ids`: conses `(gap to the next factor, ITOKey, tail id)` instead of
# the absolute site; with distance and charge it names a right vertex (`_rdesc`).
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
