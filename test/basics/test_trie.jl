using Test

using OpSum
using OpSum: Trie

@testset "Core functionality" begin
    t = Trie{Char,Int}()
    @test keytype(t) === Char
    @test valtype(t) === Int

    t["amy"] = 56
    t["ann"] = 15
    t["emma"] = 30
    t["rob"] = 27
    t["roger"] = 52
    t["kevin"] = Int8(11)

    @test length(t) == 6
    @test haskey(t, "roger")
    @test !haskey(t, "karen")

    @inferred Nothing get(t, "rob", nothing)
    @test get(t, "rob", nothing) == 27
    @test isnothing(get(t, "rod", nothing))

    @inferred Nothing t["amy"]
    @test_throws KeyError t["notamy"]

    @test eltype(t) === valtype(t)

    @test sort(keys(t)) == collect.(["amy", "ann", "emma", "kevin", "rob", "roger"])
    @test t["rob"] == 27
    t["rob"] = 2
    @test t["rob"] == 2
end
