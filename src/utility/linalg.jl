function MatrixAlgebraKit.copy_input(::typeof(svd_compact), A::SparseMatrixDOK)
    return MatrixAlgebraKit.copy_input(svd_compact, Matrix(A))
end

function SparseArraysBase.default_mul!!(C, A, B, α::Number=true, β::Number=false)
    return β * C + α * A * B
end
