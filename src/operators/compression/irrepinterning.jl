# Pure ITOTermTable interning helpers: contiguous column suffixes/prefixes, and the
# translation-invariant suffix naming, shared by every sweep backend in irrepgraph*.jl.

"""
    _suffix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Intern every term's *contiguous column suffixes*. `tt.sites` columns are ascending and zero-padded, so
a term's active factors at sites `> i` are always a suffix `j₀:K` of its column; `sufid[j, t]` is a
dense integer id for the factor list `j:K` of term `t`, with `0` for the exhausted suffix. Built
bottom-up in `Θ(M·K)` — the replacement for materialising a length-`N` path per term.

Equality of ids is equality of the remaining factor list. That is *not* by itself equality of the
suffix path: `_op_at_ito` fills idle sites with a pass-through carrying the running bond charge, and
the idle sites *before* the first remaining factor carry the charge accumulated so far. So a suffix
path is identified by the pair `(sufid[j₀, t], running bond charge)` — see `_signature`.
"""
# Shared walk/insert-on-miss bookkeeping behind `_suffix_ids`, `_prefix_ids` and `_rel_suffix_ids`.
# `dir = -1` walks columns `K:-1:1`, writing `id[j, t]` from the already-interned tail `id[j + 1, t]`
# (a suffix walk: padding only ever trails, so a zero column just has nothing to write and the walk
# continues past it); `dir = +1` walks `1:K`, writing `id[j + 1, t]` from the head `id[j, t]` and
# stopping at the first zero (every later column is padding too, so there is nothing left to break
# out of). `keyfn(j, t, s)` builds the tuple to intern from the current column and its neighbour.
function _intern_columns!(intern::Dictionary, id::Matrix{Int}, tt::ITOTermTable, M::Int, K::Int, dir::Int, keyfn)
    nid = 0
    for t in 1:M
        for j in (dir < 0 ? (K:-1:1) : (1:K))
            s = tt.sites[j, t]
            if iszero(s)
                if dir < 0
                    continue
                else
                    break
                end
            end
            trans = keyfn(j, t, s)
            v = get(intern, trans, 0)
            if iszero(v)
                nid += 1
                v = nid
                insert!(intern, trans, v)
            end
            if dir < 0
                id[j, t] = v
            else
                id[j + 1, t] = v
            end
        end
    end
    return id
end

function _suffix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    sufid = zeros(Int, K + 1, M)      # row K+1 and every padded position stay 0 == exhausted
    intern = Dictionary{Tuple{Int, ITOKey{I}, Int}, Int}()
    _intern_columns!(
        intern, sufid, tt, M, K, -1,
        (j, t, s) -> (s, tt.keys[j, t], sufid[j + 1, t])
    )
    return sufid
end

"""
    _prefix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Intern every term's *contiguous column prefixes* — the mirror image of [`_suffix_ids`](@ref).
`preid[j, t]` is a dense integer id for the factor list `1:j-1` of term `t` (so `preid[1, t] == 0`,
the empty prefix), built top-down in `Θ(M·K)`.

Equality of ids **is** equality of the prefix path `o_t[1:b]`, with no charge component needed: at
every idle site of the prefix `_op_at_ito` fills in a pass-through whose running charge is fixed by
the factors to its left, i.e. by the factor list itself. (The suffix is the asymmetric one — the
idle sites *before* its first remaining factor carry the charge accumulated by the prefix, which is
why `_signature!` pairs `sufid` with the running charge.)

Used by the `IndependentSVD` sweep, which classifies both sides of every bond from scratch.
"""
function _prefix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    preid = zeros(Int, K + 1, M)      # row 1 stays 0 == the empty prefix
    intern = Dictionary{Tuple{Int, Int, ITOKey{I}}, Int}()
    _intern_columns!(
        intern, preid, tt, M, K, 1,
        (j, t, s) -> (preid[j, t], s, tt.keys[j, t])
    )
    return preid
end

"""
    _rel_suffix_ids(tt::ITOTermTable{I}) -> Matrix{Int}

Translation-invariant twin of [`_suffix_ids`](@ref): interns each term's contiguous column suffixes by
*shape* rather than by absolute position, consing `(gap to the next active site, ITOKey, tail id)`
instead of `(absolute site, ITOKey, tail id)`.

`_suffix_ids` is what the per-bond merge wants — at a fixed bond, "same remaining factor list" and
"same remaining factor list at the same absolute sites" coincide, and the absolute form is also what
`pendbysig` keys on. Comparing suffix classes *across* bonds needs the positional information split
off instead: a class and its translate one unit cell later have the same shape and the same distance
from their respective bonds. `(distance, relsufid, running charge)` is exactly that pair of facts, and
is the canonical name of a right vertex — see [`_rdesc`](@ref).
"""
function _rel_suffix_ids(tt::ITOTermTable{I}) where {I}
    K, M = arity(tt), nterms(tt)
    relid = zeros(Int, K + 1, M)
    intern = Dictionary{Tuple{Int, ITOKey{I}, Int}, Int}()
    _intern_columns!(
        intern, relid, tt, M, K, -1,
        function (j, t, s)
            nxt = j < K ? tt.sites[j + 1, t] : 0
            gap = iszero(nxt) ? 0 : nxt - s           # 0 == this is the last factor
            return (gap, tt.keys[j, t], relid[j + 1, t])
        end
    )
    return relid
end
