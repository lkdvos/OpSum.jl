using Test

using OpSum
using OpSum.PauliOperators: X, Z

using LinearAlgebra: norm

# @testset "Simple linear algebra" begin
    XpZ = @inferred X[1] + Z[1]
    ZpX = @inferred Z[1] + X[1]
    @test XpZ ≈ ZpX
    @test 2 * X ≈ X + X ≈ X * 2
    @test XpZ - Z ≈ X
    @test norm(X + X) ≈ 2 * norm(X)
    @test norm(X + Z) ≈ norm([norm(X)^2, norm(Z)^2])
end
