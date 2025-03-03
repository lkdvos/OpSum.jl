
# compress_state_machine(Ws; kwargs...) = compress_state_machine!(copy(Ws); kwargs...)
function compress_state_machine(Ws; alg=nothing, trunc=nothing)
    @assert length(Ws) > 1 "TODO?"
    local Vleft, Vright
    return map(1:length(Ws)) do i
        i == 1 && return Ws[i]
        M = extract_bondmatrix(Ws[i])
        _, _, Vt = svd_trunc(M; alg, trunc)
        Vright = pack_bondmatrix(Vt)
        W′ = if i == 2
            Ws[i] * Vright'
        elseif i == length(Ws)
            Vleft * Ws[i]
        else
            Vleft * Ws[i] * Vright'
        end
        Vleft = Vright
        return W′
    end
end

function compress_state_machine_serial(Ws; alg=nothing, trunc=nothing)
    TW′ = Matrix{eltype(Ws)}
    # left-to-right pass
    Ws′ = similar(Ws, TW′)
    Ws′[1] = Ws[1]
    for i in 1:length(Ws)
        M = extract_bondmatrix(Ws[i])
        U, S, Vt = svd_trunc(M; alg, trunc)
        Up, Sp, Vtp = pack_bondmatrix.((U, S, Vt))
        Ws[i] = Up
    end
end

# M = W[2:(end-1), 2:(end-1)]
# but extract only coefficients
function extract_bondmatrix(W::SparseMatrixDOK)
    display(Matrix(W))
    size(W, 1) > 2 && size(W, 2) > 2 || return similar(W, scalartype(eltype(W)), (0, 0))
    M = similar(W, scalartype(eltype(W)), size(W) .- 2)
    for (key, val) in storedpairs(W)
        if key[1] != 1 && key[1] != size(W, 1) && key[2] != 1 && key[2] != size(W, 2)
            c = coefficient(val)
            M[key - CartesianIndex(1, 1)] = c
        end
    end
    @show M.storage
    @show storedpairs(W)
    return @show M
end
function extract_bondmatrix(W::Matrix)
    size(W, 1) > 2 && size(W, 2) > 2 || return similar(W, scalartype(eltype(W)), (0, 0))
    M = similar(W, scalartype(eltype(W)), size(W) .- 2)
    for I in eachindex(IndexCartesian(), W)
        if I[1] != 1 && I[1] != size(W, 1) && I[2] != 1 && I[2] != size(W, 2)
            c = coefficient(W[I])
            M[I - CartesianIndex(1, 1)] = c
        end
    end
    return M
end

function pack_bondmatrix(M)
    M1 = similar(M, (1, 1))
    M1[1] = one(eltype(M))
    return cat(M1, M, M1; dims=(1, 2))
end
