# Maximum matching + minimum vertex cover (src/datastructures/bipartite.jl). The cover size is the bond
# dimension; König's theorem (|min cover| == |max matching|) is the oracle, plus an edge-cover check.

using Test
using Random
using OpSum: hopcroft_karp, min_vertex_cover_bipartite

function covers_all_edges(adjU, coverU, coverV)
    for u in eachindex(adjU), v in adjU[u]
        (coverU[u] || coverV[v]) || return false
    end
    return true
end

function is_matching(adjU, nV, pairU, pairV, matching)
    count(!iszero, pairU) == matching || return false
    count(!iszero, pairV) == matching || return false
    for u in eachindex(pairU)
        v = pairU[u]
        iszero(v) && continue
        (1 <= v <= nV && pairV[v] == u && v in adjU[u]) || return false
    end
    return true
end

@testset "bipartite matching and cover" begin
    @testset "small hand-checked graphs" begin
        # a star: the minimum cover is the centre, never the leaves
        adjU = [[1, 2, 3, 4]]
        cU, cV, _, _, m = min_vertex_cover_bipartite(adjU, 1, 4)
        @test m == 1
        @test cU == [true]
        @test !any(cV)

        cU, cV, _, _, m = min_vertex_cover_bipartite([Int[], Int[]], 2, 3)
        @test m == 0
        @test !any(cU) && !any(cV)
    end

    @testset "long alternating path (iterative DFS depth)" begin
        # a path of length 2n-1; pins that a deep component does not overflow the stack
        n = 20_000
        adjU = [k == 1 ? [1] : [k - 1, k] for k in 1:n]
        cU, cV, pairU, pairV, matching = min_vertex_cover_bipartite(adjU, n, n)
        @test matching == n
        @test count(cU) + count(cV) == matching
        @test covers_all_edges(adjU, cU, cV)
        @test is_matching(adjU, n, pairU, pairV, matching)
    end

    @testset "random graphs: König" begin
        rng = MersenneTwister(0x00C0FFEE)
        for trial in 1:200
            n = rand(rng, 1:12)
            m = rand(rng, 1:12)
            p = rand(rng, (0.1, 0.3, 0.6, 0.9))
            adjU = [findall(<(p), rand(rng, m)) for _ in 1:n]

            cU, cV, pairU, pairV, matching = min_vertex_cover_bipartite(adjU, n, m)
            @test is_matching(adjU, m, pairU, pairV, matching)
            @test covers_all_edges(adjU, cU, cV)
            @test count(cU) + count(cV) == matching   # König's theorem

            # brute force on the smallest instances
            if n + m <= 12
                best = typemax(Int)
                for maskU in 0:(2^n - 1), maskV in 0:(2^m - 1)
                    bU = [isodd(maskU >> (u - 1)) for u in 1:n]
                    bV = [isodd(maskV >> (v - 1)) for v in 1:m]
                    covers_all_edges(adjU, bU, bV) && (best = min(best, count(bU) + count(bV)))
                end
                @test best == matching
            end
        end
    end

    # Berge: a matching is maximum iff no augmenting path exists (plain BFS, independent of `hopcroft_karp`)
    function has_augmenting_path(adjU, nV, pairU, pairV)
        seenV = falses(nV)
        queue = findall(iszero, pairU)
        head = 1
        while head <= length(queue)
            u = queue[head]
            head += 1
            for v in adjU[u]
                seenV[v] && continue
                seenV[v] = true
                iszero(pairV[v]) && return true
                push!(queue, pairV[v])
            end
        end
        return false
    end

    @testset "hopcroft_karp matching size is maximum" begin
        rng = MersenneTwister(42)
        for _ in 1:100
            n, m = rand(rng, 1:10), rand(rng, 1:10)
            adjU = [findall(<(0.4), rand(rng, m)) for _ in 1:n]
            pairU, pairV, matching = hopcroft_karp(adjU, n, m)
            @test is_matching(adjU, m, pairU, pairV, matching)
            # Berge, not König: a cover from the same call would be circular
            @test !has_augmenting_path(adjU, m, pairU, pairV)
        end
    end
end
