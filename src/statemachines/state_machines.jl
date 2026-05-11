function mpo_to_dense(Ws::Vector{<:SparseMatrixDOK{<:LocalOp}}, sites)
    N = length(Ws)
    T = ComplexF64

    # Instantiate W[1] entries into physical-space matrices
    D_left, D_right = size(Ws[1])
    d = sites[1]
    result = Matrix{Matrix{T}}(undef, D_left, D_right)
    for r in 1:D_left, c in 1:D_right
        result[r, c] = zeros(T, d, d)
    end
    for (idx, v) in storedpairs(Ws[1])
        result[idx[1], idx[2]] = T.(instantiate(v, sites[1]))
    end

    # Contract left-to-right via matrix multiplication with kron
    for i in 2:N
        D_old_right = size(result, 2)
        D_new_right = size(Ws[i], 2)
        di = sites[i]

        # Instantiate W[i]
        Wi = Matrix{Matrix{T}}(undef, D_old_right, D_new_right)
        for r in 1:D_old_right, c in 1:D_new_right
            Wi[r, c] = zeros(T, di, di)
        end
        for (idx, v) in storedpairs(Ws[i])
            Wi[idx[1], idx[2]] = T.(instantiate(v, sites[i]))
        end

        # Matrix multiply: new[r,c] = sum_k kron(result[r,k], Wi[k,c])
        phys_dim = size(result[1, 1], 1) * di
        D_old_left = size(result, 1)
        new_result = Matrix{Matrix{T}}(undef, D_old_left, D_new_right)
        for r in 1:D_old_left, c in 1:D_new_right
            new_result[r, c] = zeros(T, phys_dim, phys_dim)
            for k in 1:D_old_right
                new_result[r, c] .+= kron(result[r, k], Wi[k, c])
            end
        end
        result = new_result
    end

    return result[1, end]
end
