using OpSum
using OpSum: mpo_to_dense, instantiate
using OpSum.PauliOperators: X, Y, Z
using Test
using LinearAlgebra: norm

L = 5
vertices = 1:L
sites = fill(2, L)   # qubit (dim-2) per site

@testset "mpo_bond_optimizations — single 2-site term" begin
    J = 2.0
    H = J * X[1] * X[2]
    Ws = mpo_bond_optimizations(vertices, H)

    @test length(Ws) == L
    # Bond dimension: trivially 1 everywhere (single term)
    @test all(==(1), minimum.(size.(Ws)))

    H_mpo = mpo_to_dense(Ws, sites)
    H_dense = instantiate(H, sites)
    @test H_mpo ≈ H_dense
end

@testset "mpo_bond_optimizations — single-site terms" begin
    H = sum(Z[i] for i in 1:L)
    Ws = mpo_bond_optimizations(vertices, H)
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 2
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end


@testset "mpo_bond_optimizations — nearest-neighbour XX chain" begin
    H = sum(X[i] * X[i + 1] for i in 1:(L - 1))
    Ws = mpo_bond_optimizations(vertices, H)
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 3
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)

    H += sum(Z[i] for i in 1:L)
    Ws = mpo_bond_optimizations(vertices, H)
    @test length(Ws) == L
    @test maximum(x -> max(size(x)...), Ws) <= 3
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "mpo_bond_optimizations — Heisenberg chain" begin
    H = sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1))
    Ws = mpo_bond_optimizations(vertices, H)
    @test length(Ws) == L
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "mpo_bond_optimizations — all-to-all XX (long-range)" begin
    H = sum(X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L)
    Ws = mpo_bond_optimizations(vertices, H)
    @test length(Ws) == L
    @test mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)
end

@testset "mpo_bond_optimizations — mixed coefficients" begin
    H = 1.5 * X[1] * X[3] + 0.7 * X[2] * X[3]
    Ws = mpo_bond_optimizations(1:3, H)
    @test length(Ws) == 3
    @test mpo_to_dense(Ws, sites[1:3]) ≈ instantiate(H, sites[1:3])
end
