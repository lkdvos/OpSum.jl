# Connected components of a bipartite graph (union-find), sector-agnostic
# =========================================================================
# Splits a bipartite adjacency structure into its connected components so each can be handed to
# `min_vertex_cover_bipartite` independently, instead of solving one matching problem over the
# whole graph. A minimum vertex cover of a disjoint union of graphs is exactly the union of the
# minimum vertex covers of its components (König's theorem applies per-component), so this is a
# pure decomposition: solving each component separately reproduces the same global cover.

function _find_root!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

function _union!(parent::Vector{Int}, x::Int, y::Int)
    rx, ry = _find_root!(parent, x), _find_root!(parent, y)
    rx == ry && return nothing
    parent[rx] = ry
    return nothing
end

"""
    bipartite_connected_components(adjU::Vector{Vector{Int}}, nV::Int)
        -> (us_of_component::Vector{Vector{Int}}, vs_of_component::Vector{Vector{Int}})

Connected components of the bipartite graph with left vertices `1:length(adjU)`, right vertices
`1:nV`, and edges `u => v` for `v in adjU[u]`. Each component's `us`/`vs` are the global left/right
vertex ids belonging to it, both in ascending order. A right vertex with no incident edge is
dropped (it belongs to no component); every left vertex is assumed to have at least one edge.
"""
function bipartite_connected_components(adjU::Vector{Vector{Int}}, nV::Int)
    nU = length(adjU)
    parent = collect(1:(nU + nV))
    for u in 1:nU, v in adjU[u]
        _union!(parent, u, nU + v)
    end

    comp_of_root = Dict{Int, Int}()
    us_of_component = Vector{Int}[]
    vs_of_component = Vector{Int}[]

    for u in 1:nU
        r = _find_root!(parent, u)
        cid = get!(comp_of_root, r) do
            push!(us_of_component, Int[])
            push!(vs_of_component, Int[])
            return length(us_of_component)
        end
        push!(us_of_component[cid], u)
    end

    for v in 1:nV
        r = _find_root!(parent, nU + v)
        cid = get(comp_of_root, r, 0)
        iszero(cid) && continue # v has no incident edge: belongs to no component
        push!(vs_of_component[cid], v)
    end

    return us_of_component, vs_of_component
end
