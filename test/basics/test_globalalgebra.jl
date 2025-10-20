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
