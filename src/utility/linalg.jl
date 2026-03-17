function MatrixAlgebraKit.copy_input(::typeof(svd_compact), A::SparseMatrixDOK)
    return MatrixAlgebraKit.copy_input(svd_compact, Matrix(A))
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
