function MatrixAlgebraKit.copy_input(::typeof(svd_compact), A::SparseMatrixDOK)
    return MatrixAlgebraKit.copy_input(svd_compact, Matrix(A))
end
