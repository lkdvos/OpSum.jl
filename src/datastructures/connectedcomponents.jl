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
