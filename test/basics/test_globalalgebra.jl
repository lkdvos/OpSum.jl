using Test

using OpSum
using OpSum.PauliOperators: X, Z
using VectorInterface: inner

using LinearAlgebra: norm

@testset "Simple linear algebra" begin
    XpZ = @inferred X[1] + Z[1]
    ZpX = @inferred Z[1] + X[1]
    @test XpZ ≈ ZpX
    @test 2 * X[1] ≈ X[1] + X[1] ≈ X[1] * 2
    @test XpZ - Z[1] ≈ X[1]
    @test norm(X[1] + X[1]) ≈ 2 * norm(X[1])
    @test norm(X[1] + Z[1]) ≈ norm([norm(X[1])^2, norm(Z[1])^2])
    @test inner(X[1], X[2]) ≈ 0
end

function exponentiate(H, dt; order::Integer = 2)
    0 <= order || throw(ArgumentError("Order must be a non-negative integer."))
    return sum(n -> dt^n / factorial(n) * H, 0:order)
end

@testset "Exponentiation" begin
    L = 5
    dt = 0.1

    H_symbolic = sum(X[i] * X[i + 1] for i in 1:(L - 1)) + sum(Z[i] for i in 1:L)
    expH_symbolic = exponentiate(H_symbolic, dt, order = 2)

    H_dense = OpSum.instantiate(H_symbolic, fill(2, L))
    expH_dense = exponentiate(H_dense, dt, order = 2)

    @test expH_dense ≈ OpSum.instantiate(expH_symbolic, fill(2, L))
end
