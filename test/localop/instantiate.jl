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

@testset "instantiation" verbose = true begin
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
