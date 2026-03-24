using OpSum
using OpSum: mpo_to_dense, instantiate
using OpSum.PauliOperators: X, Y, Z
using MatrixAlgebraKit: truncrank
using Test
using LinearAlgebra: norm

L = 5
vertices = 1:L
sites = fill(2, L)

@testset "SVDBondAlgorithm — single 2-site term" begin
    J = 2.0
    H = J * X[1] * X[2]
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())

    @test length(Ws) == L
    @test all(==(1), minimum.(size.(Ws)))
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "SVDBondAlgorithm — single-site terms" begin
    H = sum(Z[i] for i in 1:L)
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 2
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)

    H = sum(rand() * Z[i] for i in 1:L)
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 2
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "SVDBondAlgorithm — nearest-neighbour XX chain" begin
    H = sum(X[i] * X[i + 1] for i in 1:(L - 1))
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 3
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)

    H += sum(Z[i] for i in 1:L)
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 3
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "SVDBondAlgorithm — Heisenberg chain" begin
    H = sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1))
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    @test length(Ws) == L
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "SVDBondAlgorithm — all-to-all XX (long-range)" begin
    H = sum(X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L)
    Ws = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    @test length(Ws) == L
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "SVDBondAlgorithm — mixed coefficients" begin
    H = 1.5 * X[1] * X[3] + 0.7 * X[2] * X[3]
    Ws = mpo_bond_optimizations(1:3, H, SVDBondAlgorithm())
    @test length(Ws) == 3
    @test mpo_to_dense(Ws, sites[1:3]) ≈ instantiate(H, sites[1:3])
end

@testset "SVDBondAlgorithm — agrees with BipartiteAlgorithm" begin
    H = sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1))
    Ws_svd = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    Ws_bip = mpo_bond_optimizations(vertices, H, BipartiteAlgorithm())
    @test mpo_to_dense(Ws_svd, sites) ≈ mpo_to_dense(Ws_bip, sites)

    H = sum(X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L)
    Ws_svd = mpo_bond_optimizations(vertices, H, SVDBondAlgorithm())
    Ws_bip = mpo_bond_optimizations(vertices, H, BipartiteAlgorithm())
    @test mpo_to_dense(Ws_svd, sites) ≈ mpo_to_dense(Ws_bip, sites)
end

@testset "SVDBondAlgorithm — truncation reduces bond dimension" begin
    # H = X[1]*X[3] + X[2]*X[4] has a rank-2 coefficient matrix at bond 2,
    # so SVD without truncation needs bond dim 2 there.
    H = X[1] * X[3] + X[2] * X[4]
    Ws_exact = mpo_bond_optimizations(1:4, H, SVDBondAlgorithm())
    Ws_trunc = mpo_bond_optimizations(1:4, H, SVDBondAlgorithm(truncrank(1)))

    @test maximum(x -> max(size(x)...), Ws_exact) == 2
    @test maximum(x -> max(size(x)...), Ws_trunc) == 1

    # Exact result is correct; truncated is an approximation (different Hamiltonian)
    H_ref   = instantiate(H, sites[1:4])
    H_exact = mpo_to_dense(Ws_exact, sites[1:4])
    H_trunc = mpo_to_dense(Ws_trunc, sites[1:4])
    @test H_exact ≈ H_ref
    @test !(H_trunc ≈ H_ref)
    @test norm(H_trunc - H_ref) > 0
end
