using Test
using OpSum
using OpSum: LocalOp, instantiate
using OpSum.PauliOperators: I, X, Y, Z
using LinearAlgebra: kron

# Reference Pauli matrices
const σI = ComplexF64[1 0; 0 1]
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

# Helper: tensor product over a list of matrices
⊗(mats) = foldl(kron, mats)
⊗(mats...) = foldl(kron, mats)

# Helper: embed single-site operator at site i in an N-site system
function embed(σ, i, N)
    ops = [j == i ? σ : σI for j in 1:N]
    return ⊗(ops)
end

# Helper: embed two-site operator (σ₁ ⊗ σ₂) at sites i,j in an N-site system
function embed2(σ₁, σ₂, i, j, N)
    ops = [k == i ? σ₁ : (k == j ? σ₂ : σI) for k in 1:N]
    return ⊗(ops)
end

@testset "instantiate - LocalOp" verbose = true begin
    @testset "single basis elements" begin
        @test instantiate(I, 2) ≈ σI
        @test instantiate(X, 2) ≈ σX
        @test instantiate(Y, 2) ≈ σY
        @test instantiate(Z, 2) ≈ σZ
    end

    @testset "scaled operators" begin
        @test instantiate(2.0 * X, 2) ≈ 2.0 * σX
        @test instantiate(1.5 * Z, 2) ≈ 1.5 * σZ
        @test instantiate(im * X, 2) ≈ im * σX
        @test instantiate(-Z, 2) ≈ -σZ
    end

    @testset "sum of operators" begin
        @test instantiate(X + Z, 2) ≈ σX + σZ
        @test instantiate(X + Y + Z, 2) ≈ σX + σY + σZ
        @test instantiate(2 * X + 3 * Z, 2) ≈ 2 * σX + 3 * σZ
        @test instantiate(X - Z, 2) ≈ σX - σZ
    end

    @testset "Kronecker product (2-site)" begin
        @test instantiate(kron(X, Z), [2, 2]) ≈ kron(σX, σZ)
        @test instantiate(kron(X, X), [2, 2]) ≈ kron(σX, σX)
        @test instantiate(kron(Y, Y), [2, 2]) ≈ kron(σY, σY)
        @test instantiate(kron(I, Z), [2, 2]) ≈ kron(σI, σZ)
        @test instantiate(kron(X, I), [2, 2]) ≈ kron(σX, σI)
    end

    @testset "Kronecker product (3-site)" begin
        @test instantiate(kron(X, kron(Y, Z)), [2, 2, 2]) ≈ kron(kron(σX, σY), σZ)
        @test instantiate(kron(kron(X, Y), Z), [2, 2, 2]) ≈ kron(kron(σX, σY), σZ)
        @test instantiate(kron(I, kron(I, Z)), [2, 2, 2]) ≈ kron(kron(σI, σI), σZ)
    end

    @testset "Kronecker product with scaling" begin
        @test instantiate(kron(2.0 * X, Z), [2, 2]) ≈ kron(2.0 * σX, σZ)
        @test instantiate(kron(X, 0.5 * Z), [2, 2]) ≈ kron(σX, 0.5 * σZ)
    end

    @testset "result type" begin
        result = instantiate(X, 2)
        @test result isa Matrix{ComplexF64}
        @test size(result) == (2, 2)

        result2 = instantiate(kron(X, Z), [2, 2])
        @test result2 isa Matrix{ComplexF64}
        @test size(result2) == (4, 4)
    end
end


@testset "instantiate — GlobalOp"  verbose = true begin
    @testset "single-site on matching system" begin
        @test OpSum.instantiate(X[1], [2]) ≈ σX
        @test OpSum.instantiate(Y[1], [2]) ≈ σY
        @test OpSum.instantiate(Z[1], [2]) ≈ σZ
    end

    @testset "single-site embedded in larger system" begin
        @test OpSum.instantiate(Z[1], [2, 2]) ≈ kron(σZ, σI)
        @test OpSum.instantiate(Z[2], [2, 2]) ≈ kron(σI, σZ)
        @test OpSum.instantiate(X[1], [2, 2, 2]) ≈ embed(σX, 1, 3)
        @test OpSum.instantiate(X[2], [2, 2, 2]) ≈ embed(σX, 2, 3)
        @test OpSum.instantiate(X[3], [2, 2, 2]) ≈ embed(σX, 3, 3)
    end

    @testset "two-site product on matching system" begin
        @test OpSum.instantiate(X[1] * X[2], [2, 2]) ≈ kron(σX, σX)
        @test OpSum.instantiate(X[1] * Z[2], [2, 2]) ≈ kron(σX, σZ)
        @test OpSum.instantiate(Y[1] * Y[2], [2, 2]) ≈ kron(σY, σY)
    end

    @testset "two-site product embedded in larger system" begin
        @test OpSum.instantiate(X[1] * X[2], [2, 2, 2]) ≈ embed2(σX, σX, 1, 2, 3)
        @test OpSum.instantiate(X[2] * X[3], [2, 2, 2]) ≈ embed2(σX, σX, 2, 3, 3)
        @test OpSum.instantiate(Z[1] * Z[3], [2, 2, 2]) ≈ embed2(σZ, σZ, 1, 3, 3)
    end

    @testset "sum of single-site operators" begin
        @test OpSum.instantiate(X[1] + Z[2], [2, 2]) ≈ kron(σX, σI) + kron(σI, σZ)
        @test OpSum.instantiate(Z[1] + Z[2] + Z[3], [2, 2, 2]) ≈
            embed(σZ, 1, 3) + embed(σZ, 2, 3) + embed(σZ, 3, 3)
    end

    @testset "scaled operators" begin
        @test OpSum.instantiate(2.0 * Z[1], [2, 2]) ≈ 2.0 * embed(σZ, 1, 2)
        @test OpSum.instantiate(-Z[2], [2, 2]) ≈ -embed(σZ, 2, 2)
        @test OpSum.instantiate(0.5 * X[1] * X[2], [2, 2]) ≈ 0.5 * kron(σX, σX)
    end

    @testset "transverse-field Ising model" begin
        N = 6
        g = 1.3
        J = 0.7
        sites = fill(2, N)

        H = sum(1:N) do i
            -g * Z[i]
        end +
            sum(1:(N - 1)) do i
            J * X[i] * X[i + 1]
        end

        H_dense = -g * sum(i -> embed(σZ, i, N), 1:N) +
            J * sum(i -> embed2(σX, σX, i, i + 1, N), 1:(N - 1))

        @test OpSum.instantiate(H, sites) ≈ H_dense
    end

    @testset "Heisenberg model" begin
        N = 4
        sites = fill(2, N)

        H = sum(1:(N - 1)) do i
            X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1]
        end

        H_dense = sum(1:(N - 1)) do i
            embed2(σX, σX, i, i + 1, N) +
                embed2(σY, σY, i, i + 1, N) +
                embed2(σZ, σZ, i, i + 1, N)
        end

        @test OpSum.instantiate(H, sites) ≈ H_dense
    end

    @testset "next-nearest-neighbor interactions" begin
        N = 5
        sites = fill(2, N)

        H = sum(1:(N - 2)) do i
            X[i] * X[i + 2]
        end

        H_dense = sum(i -> embed2(σX, σX, i, i + 2, N), 1:(N - 2))

        @test OpSum.instantiate(H, sites) ≈ H_dense
    end

    @testset "result type and size" begin
        result = OpSum.instantiate(Z[1], [2])
        @test result isa Matrix
        @test size(result) == (2, 2)

        result2 = OpSum.instantiate(X[1] * X[2], [2, 2])
        @test size(result2) == (4, 4)

        result3 = OpSum.instantiate(Z[1] + Z[2], [2, 2, 2])
        @test size(result3) == (8, 8)
    end
end
