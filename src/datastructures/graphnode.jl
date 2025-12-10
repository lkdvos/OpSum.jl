struct GraphNode{K, V}
    parents::Vector{Pair{K, GraphNode{K, V}}}
    children::Vector{Pair{K, GraphNode{K, V}}}
    value::Union{Nothing, V}

    function GraphNode{K, V}(value::Union{V, Nothing} = nothing) where {K, V}
        parents = Vector{Pair{K, GraphNode{K, V}}}()
        children = Vector{Pair{K, GraphNode{K, V}}}()
        return new{K, V}(parents, children, value)
    end
end

GraphNode(trie::Trie{K, V}) where {K, V} = GraphNode{K, V}(trie)
function GraphNode{K, V}(trie::Trie) where {K, V}
    root = GraphNode{K, V}(trie.value)
    for (k, v) in pairs(trie.children)
        v′ = GraphNode{K, V}(v)
        push!(root.children, k => v′)
        push!(v′.parents, k => root)
    end
    return root
end

edgetype(x::GraphNode) = edgetype(typeof(x))
edgetype(::Type{GraphNode{K, V}}) where {K, V} = K

function _operators(x::GraphNode)
    ops = Vector{edgetype(x)}[]
    if isempty(x.children) # leaf
        push!(ops, edgetype(x)[])
        return ops
    end
    for (k, v) in x.children
        subops = _operators(v)
        for subop in subops
            push!(ops, pushfirst!(subop, k))
        end
    end
    return ops
end

function Base.show(io::IO, x::GraphNode)
    print(io, "GraphNode(")
    ops = _operators(x)
    show(IOContext(IOContext(io, :compact => true), :typeinfo => typeof(ops)), ops)

    # show(io, x.children)
    # print(io, ", ")
    # show(io, x.value)
    return print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", x::GraphNode)
    return show(io, x)
    print(io, "GraphNode(...,")
    show(io, mime, x.children)
    print(io, ", ")
    show(io, mime, x.value)
    return print(io, ")")
end
