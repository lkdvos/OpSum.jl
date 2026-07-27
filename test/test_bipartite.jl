# Maximum matching + minimum vertex cover (src/datastructures/bipartite.jl).
#
# The MPO sweeps use `min_vertex_cover_bipartite` to pick every bond's basis, so its *size* is the
# bond dimension: an off-by-one here silently inflates every MPO. König's theorem gives an
# independent oracle — |minimum vertex cover| == |maximum matching| — which is what most of these
# tests check, on top of verifying that the returned masks really do cover every edge.

using Test
using Random
using OpSum: hopcroft_karp, min_vertex_cover_bipartite

# every edge has at least one endpoint in the cover
function covers_all_edges(adjU, coverU, coverV)
    for u in eachindex(adjU), v in adjU[u]
        (coverU[u] || coverV[v]) || return false
    end
    return true
end

# `pairU`/`pairV` are a consistent matching of the claimed size, using only real edges
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

adjmatrix(adjU, nV) = [v in adjU[u] for u in eachindex(adjU), v in 1:nV]

@testset "bipartite matching and cover" begin
    @testset "small hand-checked graphs" begin
        # a path U1-V1-U2-V2: maximum matching 2, minimum cover 2
        adjU = [[1], [1, 2]]
        cU, cV, pairU, pairV, m = min_vertex_cover_bipartite(adjU, 2, 2)
        @test m == 2
        @test count(cU) + count(cV) == m
        @test covers_all_edges(adjU, cU, cV)
        @test is_matching(adjU, 2, pairU, pairV, m)

        # a star centred on U1 (the "nothing started yet" channel of the MPO sweep): the minimum
        # cover is the single left vertex, never the leaves
        adjU = [[1, 2, 3, 4]]
        cU, cV, _, _, m = min_vertex_cover_bipartite(adjU, 1, 4)
        @test m == 1
        @test cU == [true]
        @test !any(cV)

        # an isolated left vertex is never covered
        adjU = [[1], Int[]]
        cU, cV, _, _, m = min_vertex_cover_bipartite(adjU, 2, 1)
        @test m == 1
        @test !cU[2]
        @test count(cU) + count(cV) == m

        # empty graph
        cU, cV, _, _, m = min_vertex_cover_bipartite([Int[], Int[]], 2, 3)
        @test m == 0
        @test !any(cU) && !any(cV)
    end

    @testset "complete bipartite graphs" begin
        for (n, m) in ((1, 5), (3, 3), (4, 7), (6, 2))
            adjU = [collect(1:m) for _ in 1:n]
            cU, cV, pairU, pairV, matching = min_vertex_cover_bipartite(adjU, n, m)
            @test matching == min(n, m)                    # K_{n,m}
            @test count(cU) + count(cV) == matching        # König
            @test covers_all_edges(adjU, cU, cV)
            @test is_matching(adjU, m, pairU, pairV, matching)
        end
    end

    @testset "long alternating path (iterative DFS depth)" begin
        # U_k - V_k and U_{k+1} - V_k for all k: a path of length 2n-1. The recursive DFS nested to
        # the alternating-path length here; this pins that a deep component no longer overflows.
        n = 20_000
        adjU = [k == 1 ? [1] : [k - 1, k] for k in 1:n]
        cU, cV, pairU, pairV, matching = min_vertex_cover_bipartite(adjU, n, n)
        @test matching == n
        @test count(cU) + count(cV) == matching
        @test covers_all_edges(adjU, cU, cV)
        @test is_matching(adjU, n, pairU, pairV, matching)
    end

    @testset "random graphs: König + dense/adjacency agreement" begin
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

            # the dense-matrix method must agree with the adjacency-list one
            dU, dV, _, _, dmatching = min_vertex_cover_bipartite(adjmatrix(adjU, m))
            @test dmatching == matching
            @test dU == cU
            @test dV == cV

            # brute-force minimum cover for the smallest instances
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

    # Berge: a matching is maximum iff no augmenting path exists. Independent of `hopcroft_karp` —
    # a plain BFS over alternating paths from every free left vertex.
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
                iszero(pairV[v]) && return true       # reached a free right vertex
                push!(queue, pairV[v])               # else continue along the matching edge
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
            # Berge's criterion — the real maximality certificate. Comparing against a cover from the
            # same call would be circular, since König's construction derives one from the other.
            @test !has_augmenting_path(adjU, m, pairU, pairV)
        end
    end
end
