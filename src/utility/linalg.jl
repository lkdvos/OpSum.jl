# Sparse dict-of-keys → CSC helpers
# ---------------------------------
# The bond-matrix sweeps accumulate into `Dictionary{CartesianIndex{2}, T}` and only materialise a
# sparse matrix at the very end, so all we need from `SparseArrays` is a triplet constructor and an
# iterator over stored entries.

"""
    sparse_from_dict(dict::AbstractDictionary{CartesianIndex{2}}, sz) -> SparseMatrixCSC

Materialise a dict-of-keys accumulator (`CartesianIndex(row, col) => value`) as a sparse matrix of
size `sz`. Keys are unique, so no duplicate combination happens; stored entries are kept verbatim
(including explicit zeros).
"""
function sparse_from_dict(
        dict::AbstractDictionary{CartesianIndex{2}, T}, sz::Tuple{Int, Int}
    ) where {T}
    rows = Vector{Int}(undef, length(dict))
    cols = Vector{Int}(undef, length(dict))
    vals = Vector{T}(undef, length(dict))
    for (n, (idx, v)) in enumerate(pairs(dict))
        rows[n], cols[n] = Tuple(idx)
        vals[n] = v
    end
    return sparse(rows, cols, vals, sz...)
end

"""
    storedpairs(A::SparseMatrixCSC)

Iterate the stored entries of `A` as `CartesianIndex(row, col) => value` pairs, in column-major
order. Unlike `pairs(A)` this skips structurally absent entries, which matters here because the
element type is a symbolic operator rather than a number.
"""
function storedpairs(A::SparseMatrixCSC)
    rows, vals = rowvals(A), nonzeros(A)
    return (CartesianIndex(rows[k], j) => vals[k] for j in axes(A, 2) for k in nzrange(A, j))
end

function increaseindex!(d::Dictionary, k, v)
    (found, token) = gettoken(d, k)
    if found
        settokenvalue!(d, token, gettokenvalue(d, token) + v)
    else
        insert!(d, k, v)
    end
    return d
end
