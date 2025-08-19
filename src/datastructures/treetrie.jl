mutable struct TreeTrie{K, V} <: AbstractDictionary{Tree{K}, V}
    const children::Dictionary{K, Vector{TreeTrie{K, V}}}
    value::Union{V, Nothing}

    function TreeTrie{K, V}(value::Union{V, Nothing} = nothing) where {K, V}
        children = Dictionary{K, Vector{TreeTrie{K, V}}}()
        return new{K, V}(children, value)
    end
end
