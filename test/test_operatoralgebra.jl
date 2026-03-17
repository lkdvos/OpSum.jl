using Test
using OpSum.PauliOperators: I, X, Z
using LinearAlgebra: norm

@testset "Simple linear algebra" begin
    XpZ = @inferred X + Z
    ZpX = @inferred Z + X
    @test XpZ ≈ ZpX
    @test 2 * X ≈ X + X ≈ X * 2
    @test XpZ - Z ≈ X
    @test norm(X + X) ≈ 2 * norm(X)
    @test norm(X + Z) ≈ norm([norm(X)^2, norm(Z)^2])
end
