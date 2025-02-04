struct ParallelDecomposition{F,T}
    factorization_alg::F
    truncation_alg::T
end

# TODO: manual blockmatrix or simply diag(I, Vt, I)
# TODO: right-to-left?

function compress_state_machine(Ws, alg::ParallelDecomposition)
    @assert length(Ws) > 1 "TODO?"

    # compute isometries
    basis_transforms = map(1:(length(Ws) - 1)) do i
        M = W[i][2:(end - 1), 2:(end - 1)]
        _, _, Vt = svd_trunc!(M, alg.factorization_alg, alg.truncation_alg)
        return Vt
    end

    # apply isometries
    return map(1:length(Ws)) do i
        i == 1 && return Ws[i] * Vs[i + 1]'
        i == length(Ws) && return Vs[i] * Ws[i]
        return Vs[i] * Ws[i] * Vs[i=1]'
    end
end
