function compress_vertex_operators(Ws, Ms; trunc = nothing)
    length(Ws) == 1 && return copy.(Ws)

    Vs = map(Ms) do M
        return trunc_bondcoefficient(M; trunc)
    end

    return map(enumerate(Ws)) do (i, W)
        # TODO: this should be kept sparse
        W′ = if i == 1
            W * Vs[1]'
        elseif i == length(Ws)
            Vs[end] * W
        else
            Vs[i - 1] * W * Vs[i]'
        end

        W_simplified = similar(W, axes(W′))
        return simplify_vertexoperator!(W_simplified, W′)
    end
end

function trunc_bondcoefficient(M; trunc = nothing)
    _, _, Vᴴpart = svd_trunc!(Matrix(M)[2:(end - 1), 2:(end - 1)]; trunc)
    return cat(M[1, 1], Vᴴpart, M[end, end]; dims = (1, 2))
end

function simplify_vertexoperator!(W′, W)
    for (k, v) in storedpairs(W)
        v′ = simplify(v)
        if !iszero(v′)
            W′[k] = v′
        end
    end
    return W′
end
