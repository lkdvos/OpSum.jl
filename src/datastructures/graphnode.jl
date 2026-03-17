mutable struct GraphNode{EdgeType, ValType}
    const parents::Vector{Pair{EdgeType, GraphNode{EdgeType, ValType}}}
    const children::Vector{Pair{EdgeType, Vector{GraphNode{EdgeType, ValType}}}}
    value::Union{Nothing, ValType}

    function GraphNode{E, T}(value::Union{T, Nothing} = nothing) where {E, T}
        GT = GraphNode{E, T}
        return new{E, T}(Vector{Pair{E, GT}}(), Vector{Pair{E, Vector{GT}}}(), value)
    end
end

edgetype(x::GraphNode) = edgetype(typeof(x))
edgetype(::Type{GraphNode{E, T}}) where {E, T} = E

Base.valtype(x::GraphNode) = valtype(typeof(x))
Base.valtype(::Type{GraphNode{E, T}}) where {E, T} = T

# add_neighbour!(x::GraphNode, id) = (insert!(x.neighbours, id, Vector{Pair{edgetype(x), typeof(x)}}()); x)
# get_neighbour!(x::GraphNode, id) = get!(x.neighbours, id, Vector{Pair{edgetype(x), typeof(x)}}())


function _operators(x::GraphNode, parent_id = 1)
    ops = Vector{typeof(x)}()

    ops = Vector{edgetype(x)}[]
    if length(x.neighbours) <= 1 # single node or leaf
        push!(ops, edgetype(x)[])
        return ops
    end

    # for (vertex, node) in x.neighbours
    #     vertex == parent_id && continue


    for (k, v) in x.children
        subops = _operators(v)
        for subop in subops
            push!(ops, pushfirst!(subop, k))
        end
    end
    return ops
end

# function Base.show(io::IO, x::GraphNode)
#     print(io, "GraphNode(")
#     ops = _operators(x)
#     show(IOContext(IOContext(io, :compact => true), :typeinfo => typeof(ops)), ops)

#     # show(io, x.children)
#     # print(io, ", ")
#     # show(io, x.value)
#     return print(io, ")")
# end

function Base.show(io::IO, ::MIME"text/plain", x::GraphNode)
    println(io, "GraphNode with value ", x.value)
    println(io, "parents: ")
    for (o, _) in x.parents
        println(io, '\t', o, " => …")
    end
    println(io, "children: ")
    for (o, _) in x.children
        println(io, '\t', o, " => …")
    end

    return nothing
end
