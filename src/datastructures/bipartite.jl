# Maximum matching (Hopcroft–Karp) + minimum vertex cover (König) on bipartite graphs, used by the
# sweeps to choose each bond's basis. Everything is driven by adjacency lists (`adjU[u]` lists the
# right vertices of left vertex `u`), so it is `O(E√V)` and never forms an `nU × nV` object.

# BFS: layer the alternating-path graph from all free U's. Returns the length of the *shortest*
# augmenting path (`typemax` if none). Layering stops there: deeper layers would let the DFS follow
# non-shortest paths, which breaks the O(√V) phase bound.
function hopcroft_karp_bfs!(dist, adjU, pairU, pairV, queue)
    INF = typemax(eltype(dist))
    fill!(dist, INF)
    empty!(queue)

    for u in eachindex(pairU)
        if iszero(pairU[u])
            dist[u] = 0
            push!(queue, u)
        end
    end

    found = INF
    head = 1
    while head <= length(queue)
        u = queue[head]
        head += 1
        # skips unlayered (`INF`) and past-the-shortest-layer vertices; keeps `dist[u] + 1` from overflowing
        dist[u] < found || continue
        d = dist[u] + 1
        for v in adjU[u]
            u2 = pairV[v]
            if iszero(u2)
                found = min(found, d) # reached a free V vertex: an augmenting path of length `d`
            elseif dist[u2] == INF
                dist[u2] = d
                push!(queue, u2)
            end
        end
    end

    return found
end

# DFS: augment along a path of exactly length `found` from `u0` within the layered graph; returns
# whether one was found. Iterative (explicit stack + cursor into `adjU`): the recursive form would
# nest to the path length, `O(nU)` on the large components of long-range models.
function hopcroft_karp_dfs!(adjU, pairU, pairV, dist, cursor, stack, u0, found)
    INF = typemax(eltype(dist))
    empty!(stack)
    push!(stack, u0)
    cursor[u0] = 1

    while !isempty(stack)
        u = stack[end]
        adj = adjU[u]
        k = cursor[u]
        if k > length(adj)
            dist[u] = INF # dead end: drop `u` from the layered graph for this phase
            pop!(stack)
            continue
        end
        cursor[u] = k + 1

        v = adj[k]
        u2 = pairV[v]
        if iszero(u2)
            # `v` is free: take it only if it completes a *shortest* path, then rewire the whole
            # alternating path on the stack (each `u` takes the right vertex its child was matched on).
            dist[u] + 1 == found || continue
            while true
                uu = pop!(stack)
                prev = pairU[uu]
                pairU[uu] = v
                pairV[v] = uu
                isempty(stack) && break
                v = prev
            end
            return true
        elseif dist[u2] == dist[u] + 1
            push!(stack, u2)
            cursor[u2] = 1
        end
    end

    return false
end

"""
    hopcroft_karp(adjU, nU, nV)

Maximum matching of the bipartite graph with left vertices `1:nU`, right vertices `1:nV` and
neighbour lists `adjU`, by Hopcroft–Karp in `O(E√V)`. Returns `(pairU, pairV, matching_size)`, where
`pairU[u]` / `pairV[v]` is the partner or `0` if free.
"""
function hopcroft_karp(adjU::Vector{Vector{Int}}, nU::Int, nV::Int)
    pairU = zeros(Int, nU)
    pairV = zeros(Int, nV)
    dist = zeros(Int, nU)  # BFS layer of each U vertex (INF = not in the layered graph)
    queue = Int[]
    cursor = zeros(Int, nU)
    stack = Int[]

    matching = 0
    while true
        found = hopcroft_karp_bfs!(dist, adjU, pairU, pairV, queue)
        found == typemax(eltype(dist)) && break
        for u in 1:nU
            iszero(pairU[u]) || continue
            hopcroft_karp_dfs!(adjU, pairU, pairV, dist, cursor, stack, u, found) &&
                (matching += 1)
        end
    end

    return pairU, pairV, matching
end

"""
    min_vertex_cover_bipartite(adjU::Vector{Vector{Int}}, nU, nV)
    min_vertex_cover_bipartite(A::AbstractMatrix{<:Real})

Minimum vertex cover of a bipartite graph, via a Hopcroft–Karp maximum matching plus König's
construction. Returns `(coverU, coverV, pairU, pairV, matching_size)` where `coverU`/`coverV` are
boolean masks over the left/right vertices.

The matrix method builds the adjacency lists (any nonzero is an edge) and forwards. Left vertices
with no incident edge are never covered.
"""
function min_vertex_cover_bipartite(adjU::Vector{Vector{Int}}, nU::Int, nV::Int)
    pairU, pairV, matching = hopcroft_karp(adjU, nU, nV)

    # Vertices reachable by alternating paths from unmatched U's (non-matching edges U → V, matching
    # edges V → U).
    visitedU = falses(nU)
    visitedV = falses(nV)

    stack = Int[]
    for u in 1:nU
        if iszero(pairU[u])
            push!(stack, u)
            visitedU[u] = true
        end
    end

    while !isempty(stack)
        u = pop!(stack)
        for v in adjU[u]
            (pairU[u] == v || visitedV[v]) && continue
            visitedV[v] = true
            u2 = pairV[v]
            if !iszero(u2) && !visitedU[u2]
                visitedU[u2] = true
                push!(stack, u2)
            end
        end
    end

    # König: cover = (U \ Z_L) ∪ Z_R
    return .!visitedU, visitedV, pairU, pairV, matching
end

function min_vertex_cover_bipartite(A::AbstractMatrix{<:Real})
    n, m = size(A)
    adjU = [Int[] for _ in 1:n]
    # column-major traversal leaves every `adjU[u]` ascending
    @inbounds for v in 1:m, u in 1:n
        iszero(A[u, v]) || push!(adjU[u], v)
    end
    return min_vertex_cover_bipartite(adjU, n, m)
end
