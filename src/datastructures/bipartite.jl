"""
    hopcroft_karp(adjU, nU, nV)

Compute a maximum matching in a bipartite graph using the Hopcroft–Karp algorithm.

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
    # pairU[u] = matched neighbor v in V, or 0 if free
    pairU = zeros(Int, nU)
    # pairV[v] = matched neighbor u in U, or 0 if free
    pairV = zeros(Int, nV)

    # dist[u] = distance label from BFS layering (only for U side)
    dist = fill(0, nU)

    # "infinity" value for dist
    INF = typemax(Int)

    # -------------------------------------------------------------------------
    # BFS: build layered graph of alternating paths, starting from all free U's.
    #
    # Returns true if there is at least one free vertex in V reachable
    # via an alternating path (i.e., there exists an augmenting path).
    # -------------------------------------------------------------------------
    function bfs()::Bool
        queue = Int[]

        # Initialize distances:
        #   free vertices in U start at distance 0
        #   matched vertices start at INF
        for u in 1:nU
            if pairU[u] == 0
                dist[u] = 0
                push!(queue, u)
            else
                dist[u] = INF
            end
        end

        found_augmenting = false

        # Standard BFS
        while !isempty(queue)
            u = popfirst!(queue)
            # Only proceed if this vertex is still in the layered graph
            if dist[u] < INF
                for v in adjU[u]
                    u2 = pairV[v]  # if v is matched, u2 is its partner in U
                    if u2 == 0
                        # We reached a free vertex on V side:
                        # this means there is at least one augmenting path
                        found_augmenting = true
                    elseif dist[u2] == INF
                        # If u2 has not been assigned a layer yet,
                        # put it in the next layer.
                        dist[u2] = dist[u] + 1
                        push!(queue, u2)
                    end
                end
            end
        end

        return found_augmenting
    end

    # -------------------------------------------------------------------------
    # DFS: search for an augmenting path from vertex u in U,
    #      constrained to layered graph built by BFS.
    #
    # Returns true if an augmenting path was found and matching updated.
    # -------------------------------------------------------------------------
    function dfs(u::Int)::Bool
        for v in adjU[u]
            u2 = pairV[v]
            # Case 1: v is free → we can extend the alternating path
            if u2 == 0
                pairU[u] = v
                pairV[v] = u
                return true

                # Case 2: v is matched, but we may be able to go further
            elseif dist[u2] == dist[u] + 1 && dfs(u2)
                pairU[u] = v
                pairV[v] = u
                return true
            end
        end

        # If we fail to find an augmenting path from u,
        # remove u from the layered graph for this phase.
        dist[u] = INF
        return false
    end

    # -------------------------------------------------------------------------
    # Main phase loop
    # -------------------------------------------------------------------------
    matching = 0
    while bfs()
        # Try to find augmenting paths from each free vertex in U
        for u in 1:nU
            if pairU[u] == 0
                if dfs(u)
                    matching += 1
                end
            end
        end
    end

    return pairU, pairV, matching
end

function min_vertex_cover_bipartite(A::AbstractMatrix{<:Real})
    n, m = size(A)

    adjU = [findall(!iszero, A[u, :]) for u in 1:n]
    pairU, pairV, matching = hopcroft_karp(adjU, n, m)

    # Step 1: find all vertices reachable by alternating paths
    # from unmatched vertices in U.
    visitedU = falses(n)
    visitedV = falses(m)

    # Start from all free vertices in U
    stack = Int[]
    for u in 1:n
        if pairU[u] == 0
            push!(stack, u)
            visitedU[u] = true
        end
    end

    while !isempty(stack)
        u = pop!(stack)

        # From u (in U, reachable), go via unmatched edges to V
        for v in 1:m
            if A[u, v] != 0 && !visitedV[v] && pairU[u] != v
                # edge u-v is either unmatched or we don't care about matched direction here
                visitedV[v] = true
                # From v in V, follow matched edges back to U (if any)
                u2 = pairV[v]
                if u2 != 0 && !visitedU[u2]
                    visitedU[u2] = true
                    push!(stack, u2)
                end
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
