# Connected components of a bipartite graph (union-find). A minimum vertex cover of a disjoint union
# is the union of the components' minimum covers (König per component), so splitting is a pure
# decomposition that gives `min_vertex_cover_bipartite` smaller problems.

function _find_root!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
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
