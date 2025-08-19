function compress_vertex_operators(Ws, Ms; trunc = nothing)
    length(Ws) == 1 && return copy.(Ws)

    @show Us = map(Ms) do M
        return trunc_bondcoefficient(M; trunc)
    end

    return map(enumerate(Ws)) do (i, W)
        W′ = if i == 1
            W * Us[1]
        elseif i == length(Ws)
            Us[end]' * W
        else
            Us[i - 1]' * W * Us[i]
        end
        @info "test $i" W′ QuantumOperatorAlgebra._simplify.(W′)
        return QuantumOperatorAlgebra._simplify.(W′)
    end
end


function trunc_bondcoefficient(M; trunc=nothing)
    Upart, = svd_trunc!(Matrix(M)[2:end-1,2:end-1]; trunc)
    return if length(Upart) == 0
        cat(M[1, 1], one(eltype(M)), M[end, end]; dims=(1, 2))
    else
        cat(M[1, 1], Upart, M[end, end]; dims=(1, 2))
    end
end
