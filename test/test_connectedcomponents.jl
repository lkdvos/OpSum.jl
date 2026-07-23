using Test
using OpSum: bipartite_connected_components

@testset "bipartite_connected_components" begin
    @testset "single connected component" begin
        adjU = [[1, 2], [2, 3]]
        us, vs = bipartite_connected_components(adjU, 3)
        @test length(us) == 1
        @test only(us) == [1, 2]
        @test only(vs) == [1, 2, 3]
    end

    @testset "two disjoint components" begin
        # component A: U1-V1, U1-V2, U2-V2, U3-V2 (all linked through V2) ; component B: U4-V3
        adjU = [[1, 2], [2], [2], [3]]
        us, vs = bipartite_connected_components(adjU, 3)
        @test length(us) == 2

        idxA = findfirst(u -> 4 ∉ u, us)
        idxB = findfirst(u -> 4 ∈ u, us)
        @test !isnothing(idxA) && !isnothing(idxB) && idxA != idxB
        @test us[idxA] == [1, 2, 3]
        @test vs[idxA] == [1, 2]
        @test us[idxB] == [4]
        @test vs[idxB] == [3]
    end

    @testset "right vertex with no incident edge is dropped" begin
        adjU = [[1]]
        us, vs = bipartite_connected_components(adjU, 3)
        @test length(us) == 1
        @test only(us) == [1]
        @test only(vs) == [1]     # vertices 2, 3 have no edges and belong to no component
    end

    @testset "no left vertices" begin
        us, vs = bipartite_connected_components(Vector{Int}[], 0)
        @test isempty(us)
        @test isempty(vs)
    end
end
