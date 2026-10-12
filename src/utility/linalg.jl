# The sweeps accumulate into `Dictionary{CartesianIndex{2}, T}` and materialise a sparse matrix at
# the end (entries kept verbatim, including explicit zeros).
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

# stored entries of `A` as `CartesianIndex(row, col) => value`, column-major (unlike `pairs(A)`)
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
