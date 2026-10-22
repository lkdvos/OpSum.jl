# Hopcroft–Karp matching + König minimum vertex cover, driven by adjacency lists (`adjU[u]` = right
# vertices of left vertex `u`): `O(E√V)`.

# BFS layering from the free U's; returns the shortest augmenting path length (`typemax` if none).
# Layering stops there, else the DFS could follow non-shortest paths and break the O(√V) phase bound.
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

# Augment along a path of exactly length `found` from `u0` in the layered graph. Iterative: the
# recursive form nests to the path length, `O(nU)` on long-range models.
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

# Maximum matching: `(pairU, pairV, matching_size)`, `0` marking a free vertex.
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

# `(coverU, coverV)`, boolean masks of a minimum vertex cover (König).
function minimum_vertex_cover(adjU::Vector{Vector{Int}}, nU::Int, nV::Int)
    pairU, pairV, _ = hopcroft_karp(adjU, nU, nV)

    # reachable by alternating paths from unmatched U's
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
    return .!visitedU, visitedV
end

# Connected components of a bipartite graph (union-find); a minimum cover is the union of the
# components' covers.

function _find_root!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

# `(us, vs)` per component: global left/right ids in ascending order. A right vertex with no edge
# belongs to no component; every left vertex is assumed to have an edge.
function connected_components(adjU::Vector{Vector{Int}}, nV::Int)
    nU = length(adjU)
    parent = collect(1:(nU + nV))
    for u in 1:nU, v in adjU[u]
        parent[_find_root!(parent, u)] = _find_root!(parent, nU + v)
    end

    comp_of_root = Dict{Int, Int}()
    us_of_component = Vector{Int}[]
    vs_of_component = Vector{Int}[]
    for u in 1:nU
        cid = get!(comp_of_root, _find_root!(parent, u)) do
            push!(us_of_component, Int[])
            push!(vs_of_component, Int[])
            return length(us_of_component)
        end
        push!(us_of_component[cid], u)
    end
    for v in 1:nV
        cid = get(comp_of_root, _find_root!(parent, nU + v), 0)
        iszero(cid) || push!(vs_of_component[cid], v)
    end
    return us_of_component, vs_of_component
end
