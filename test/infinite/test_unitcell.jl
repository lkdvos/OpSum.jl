using Test
using TensorKit
using LinearAlgebra: dot
using OpSum
using OpSum: window_terms, unitcell_terms, translate, maxspan, termspan, opsum

include(joinpath(@__DIR__, "..", "testutils.jl"))   # unitbasis, spinvector

const BSU2 = unitbasis(SU2Space(1 // 2 => 1))

@testset "translate / unitcell_terms / maxspan" begin
    S = spinvector(BSU2)
    H = dot(S[3], S[5])
    @test only(translate(H, 4)).sites == [7, 9]
    @test translate(H, 0) === H
    @test maxspan(H) == 2
    @test termspan(only(H)) == 2
    @test maxspan(S[1]) == 0

    # anchoring shifts the leftmost site into 1:L, by a multiple of L only
    @test only(unitcell_terms(H, 1)).sites == [1, 3]
    @test only(unitcell_terms(translate(H, 1), 2)).sites == [2, 4]
    @test only(unitcell_terms(translate(H, -6), 3)).sites == [3, 5]

    # window_terms keeps exactly the translates that fit
    chain = InfiniteChain([BSU2])
    gen = unitcell_terms(opsum(chain, dot(S[1], S[2])))
    for nc in 1:5
        w = window_terms(gen, nc)
        @test length(w) == max(0, nc - 1)
        @test all(t -> issubset(t.sites, 1:nc), w)
    end
end
