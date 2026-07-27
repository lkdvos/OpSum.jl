# Maximum matching (Hopcroft–Karp) + minimum vertex cover (König) on bipartite graphs
# ===================================================================================
# The primitive the per-bond-sector MPO sweeps use to choose each bond's basis. Everything here is
# driven by **adjacency lists**: `adjU[u]` lists the right vertices incident to left vertex `u`, so
# the whole computation is `O(E√V)` and never touches an `nU × nV` object. The dense-matrix method
# of `min_vertex_cover_bipartite` is a thin wrapper for callers that only have an adjacency matrix.

# -------------------------------------------------------------------------
# BFS: build the layered graph of alternating paths, starting from all free U's.
#
# Returns the length of the *shortest* augmenting path (`typemax` if none exists). Layering stops at
# that length: expanding deeper layers would let the DFS follow non-shortest augmenting paths, which
# is what breaks Hopcroft–Karp's O(√V) phase bound.
# -------------------------------------------------------------------------
function hopcroft_karp_bfs!(dist, adjU, pairU, pairV, queue)
    INF = typemax(eltype(dist))
    fill!(dist, INF)
    empty!(queue)

    # free vertices in U start at distance 0; matched ones stay at INF until layered
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
        # `dist[u] == INF` (not layered) and `dist[u] >= found` (past the shortest layer) both stop
        # here; the guard also keeps `dist[u] + 1` from overflowing.
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

# -------------------------------------------------------------------------
# DFS: search for an augmenting path from `u0`, constrained to the layered graph and to paths of
# exactly length `found`. Iterative (explicit stack + per-vertex cursor into `adjU`) — the recursive
# form would nest to the alternating-path length, which is `O(nU)` on the large components the
# long-range models produce.
#
# Returns true if an augmenting path was found and the matching updated.
# -------------------------------------------------------------------------
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
            # `v` is free. Only take it if this completes a *shortest* augmenting path, then rewire
            # the whole alternating path held on the stack: each `u` takes the right vertex its
            # child was matched on, from the top down.
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

Compute a maximum matching in a bipartite graph using the Hopcroft–Karp algorithm, in `O(E√V)`.

Arguments
---------
- `adjU::Vector{Vector{Int}}`:
    Adjacency list for the left part U.
    `adjU[u]` is a list of neighbors v in the right part V (1-based indices).
- `nU::Int`: number of vertices on the left side (U = 1:nU)
- `nV::Int`: number of vertices on the right side (V = 1:nV)

Returns
-------
- `pairU::Vector{Int}`: size nU, `pairU[u]` is the matched v in V or 0 if free.
- `pairV::Vector{Int}`: size nV, `pairV[v]` is the matched u in U or 0 if free.
- `matching_size::Int`: size of the maximum matching.
"""
function hopcroft_karp(adjU::Vector{Vector{Int}}, nU::Int, nV::Int)
    pairU = zeros(Int, nU) # pairU[u] = matched neighbor v in V, or 0 if free
    pairV = zeros(Int, nV) # pairV[v] = matched neighbor u in U, or 0 if free

    dist = fill(0, nU)     # BFS layer of each U vertex (INF = not in the layered graph)
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

The adjacency-list method is the primary one and runs in `O(E√V)`; the matrix method builds the
adjacency lists (treating any nonzero as an edge) and forwards. Left vertices with no incident edge
are never covered.
"""
function min_vertex_cover_bipartite(adjU::Vector{Vector{Int}}, nU::Int, nV::Int)
    pairU, pairV, matching = hopcroft_karp(adjU, nU, nV)

    # Step 1: find all vertices reachable by alternating paths from unmatched vertices in U —
    # non-matching edges U → V, matching edges V → U.
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

    # Step 2: build the minimum vertex cover:
    # coverU = U \ Z_L  => not visitedU
    # coverV = Z_R      => visitedV
    coverU = .!visitedU
    coverV = visitedV

    return coverU, coverV, pairU, pairV, matching
end

function min_vertex_cover_bipartite(A::AbstractMatrix{<:Real})
    n, m = size(A)
    adjU = [Int[] for _ in 1:n]
    # column-major traversal: cache-friendly on `A`, and leaves every `adjU[u]` ascending
    @inbounds for v in 1:m, u in 1:n
        iszero(A[u, v]) || push!(adjU[u], v)
    end
    return min_vertex_cover_bipartite(adjU, n, m)
end
