using OpSum
using OpSum: mpo_to_dense, mpo_to_opsum, opsum_vertex_operators, compress_vertex_operators,
    simplify, LocalOp, GlobalOp, SiteOp, instantiate
using OpSum.PauliOperators: I, X, Y, Z
using Test
using MatrixAlgebraKit: trunctol
using SparseArraysBase: SparseMatrixDOK
using LightSumTypes: variant
using LinearAlgebra: norm, kron

# Pauli matrices for reference
const σI = ComplexF64[1 0; 0 1]
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

@testset "simplify — LocalOp" begin
    @test simplify(X) == X
    @test simplify(Z) == Z
    @test simplify(I) == I

    # Zero-term removal: 0*X + Z → Z
    @test simplify(0 * X + Z) == Z

    # Single-term collapse with unit coefficient
    @test simplify(0 * X + 1 * Z) == Z

    # Non-unit coefficient preserved
    @test simplify(0 * X + 3 * Z) ≈ 3 * Z

    # Full zero: 0*X + 0*Z → zero
    s = simplify(0 * X + 0 * Z)
    @test iszero(s)

    # Pow with exponent 1 collapses
    pow_x = LocalOp(OpSum.Pow{typeof(X)}(X, 1))
    @test simplify(pow_x) == X

    # Pow with exponent > 1 preserved
    pow_x2 = LocalOp(OpSum.Pow{typeof(X)}(X, 2))
    s2 = simplify(pow_x2)
    @test variant(s2) isa OpSum.Pow
end

@testset "simplify — GlobalOp" begin
    @test simplify(0 * X[1] + Z[2]) ≈ Z[2]
    @test norm(simplify(0 * X[1] + 0 * Z[2])) ≈ 0 atol = 1.0e-15

    # Non-trivial sum preserved
    s = simplify(X[1] + Z[2])
    @test s ≈ X[1] + Z[2]
end

@testset "mpo_to_dense — hand-built MPOs" begin
    # 1-site MPO: 1×1 matrix with Z entry → should give σZ
    TW = typeof(Z)
    W1 = SparseMatrixDOK{TW}(undef, (1, 1))
    W1[1, 1] = Z
    sites = [2]
    result = mpo_to_dense([W1], sites)
    @test result ≈ σZ

    # 2-site identity MPO
    W_id1 = SparseMatrixDOK{TW}(undef, (1, 1))
    W_id1[1, 1] = I
    W_id2 = SparseMatrixDOK{TW}(undef, (1, 1))
    W_id2[1, 1] = I
    sites2 = [2, 2]
    result2 = mpo_to_dense([W_id1, W_id2], sites2)
    @test result2 ≈ kron(σI, σI)
end

# NOTE: opsum_vertex_operators returns (Ws, Ms) where Ms holds bond coefficients.
# The raw Ws alone only represent the full Hamiltonian when all coefficients are unity.
# For non-unit coefficients, compression (which applies Ms into Ws) is needed first.

@testset "Single-site operators (unit coeff)" begin
    for N in [2, 3, 5]
        vertices = 1:N
        sites = fill(2, N)

        H = sum(vertices) do i
            return Z[i]
        end

        Ws, Ms = opsum_vertex_operators(vertices, H)
        @test length(Ws) == N
        @test length(Ms) == N - 1

        H_dense = ComplexF64.(instantiate(H, sites))
        H_mpo = mpo_to_dense(Ws, sites)
        @test H_mpo ≈ H_dense
    end
end

@testset "Nearest-neighbor: XX chain (unit coeff)" begin
    N = 5
    vertices = 1:N
    sites = fill(2, N)

    H = sum(vertices[1:(end - 1)]) do i
        return X[i] * X[i + 1]
    end

    Ws, Ms = opsum_vertex_operators(vertices, H)
    H_dense = ComplexF64.(instantiate(H, sites))
    H_mpo = mpo_to_dense(Ws, sites)
    @test H_mpo ≈ H_dense
end

@testset "Nearest-neighbor: Heisenberg (unit coeff)" begin
    N = 4
    vertices = 1:N
    sites = fill(2, N)

    H = sum(vertices[1:(end - 1)]) do i
        return X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1]
    end

    Ws, Ms = opsum_vertex_operators(vertices, H)
    H_dense = ComplexF64.(instantiate(H, sites))
    H_mpo = mpo_to_dense(Ws, sites)
    @test H_mpo ≈ H_dense
end

@testset "Mixed XYZ terms (unit coeff)" begin
    N = 3
    vertices = 1:N
    sites = fill(2, N)

    H = X[1] * Y[2] + Y[2] * Z[3]
    Ws, Ms = opsum_vertex_operators(vertices, H)
    H_dense = instantiate(H, sites)
    H_mpo = mpo_to_dense(Ws, sites)
    @test H_mpo ≈ H_dense
end

@testset "Single-site system" begin
    H1 = Z[1]
    Ws1, Ms1 = opsum_vertex_operators(1:1, H1)
    @test length(Ws1) == 1
    @test length(Ms1) == 0
    H1_dense = ComplexF64.(instantiate(H1, [2]))
    H1_mpo = mpo_to_dense(Ws1, [2])
    @test H1_mpo ≈ H1_dense
end

@testset "Compression: Transverse-field Ising" begin
    N = 5
    vertices = 1:N
    sites = fill(2, N)
    g = 1.3
    J = 0.7

    H = sum(vertices) do i
        return -g * Z[i]
    end + sum(vertices[1:(end - 1)]) do i
        return J * X[i] * X[i + 1]
    end

    Ws, Ms = opsum_vertex_operators(vertices, H)
    Ws_c = compress_vertex_operators(Ws, Ms; trunc = trunctol(; atol = 1.0e-14))

    H_dense = ComplexF64.(instantiate(H, sites))
    H_compressed = mpo_to_dense(Ws_c, sites)
    @test H_compressed ≈ H_dense

    # Bond dimensions should not grow after compression
    for (W, Wc) in zip(Ws, Ws_c)
        @test all(size(Wc) .<= size(W))
    end
end

@testset "Compression: J1-J2 model" begin
    N = 6
    vertices = 1:N
    sites = fill(2, N)
    J1 = 1.0
    J2 = 0.5

    H = J1 * sum(vertices[1:(end - 1)]) do i
        return X[i] * X[i + 1]
    end + J2 * sum(vertices[1:(end - 2)]) do i
        return X[i] * X[i + 2]
    end

    Ws, Ms = opsum_vertex_operators(vertices, H)
    Ws_c = compress_vertex_operators(Ws, Ms; trunc = trunctol(1.0e-14))

    H_dense = ComplexF64.(instantiate(H, sites))
    H_compressed = mpo_to_dense(Ws_c, sites)
    @test H_compressed ≈ H_dense
end

@testset "Compression: Heisenberg with coefficients" begin
    N = 4
    vertices = 1:N
    sites = fill(2, N)

    Jx = 1.0
    Jy = 0.8
    Jz = 1.2
    H = sum(vertices[1:(end - 1)]) do i
        return Jx * X[i] * X[i + 1] + Jy * Y[i] * Y[i + 1] + Jz * Z[i] * Z[i + 1]
    end

    Ws, Ms = opsum_vertex_operators(vertices, H)
    Ws_c = compress_vertex_operators(Ws, Ms; trunc = trunctol(1.0e-14))

    H_dense = ComplexF64.(instantiate(H, sites))
    H_compressed = mpo_to_dense(Ws_c, sites)
    @test H_compressed ≈ H_dense
end

