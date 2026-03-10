# equal word length trie
mutable struct Trie{K, V} <: AbstractDictionary{Vector{K}, V}
    const children::Dictionary{K, Trie{K, V}}
    value::Union{V, Nothing}

    function Trie{K, V}(value::Union{V, Nothing} = nothing) where {K, V}
        children = Dictionary{K, Trie{K, V}}()
        return new{K, V}(children, value)
    end
end

# Constructors
# ------------
Trie() = Trie{Any, Any}()
function Trie(ks, vs)
    K = eltype(eltype(ks)) # K is the type of iterating the key
    V = eltype(vs)
    return Trie{K, V}(ks, vs)
end
function Trie{K, V}(ks, vs) where {K, V}
    trie = Trie{K, V}()
    for (k, v) in zip(ks, vs)
        trie[k] = v
    end
    return trie
end

Base.similar(t::Trie) = Trie{keytype(t), valtype(t)}()

# Properties
# ----------
# Base.keytype(trie::Trie) = keytype(typeof(trie))
# Base.keytype(::Type{<:Trie{K}}) where {K} = K
# Base.valtype(trie::Trie) = valtype(typeof(trie))
# Base.valtype(::Type{<:Trie{K}}) where {K,V} = V

AbstractTrees.children(trie::Trie) = keys(trie.children)
AbstractTrees.nodevalue(trie::Trie) = trie.value

function depth(trie::Trie)
    return isempty(trie.children) ? 0 : maximum(depth, trie.children; init = 0) + 1
end

# Accessors
# ---------
function Base.haskey(trie, key)
    subtrie = trie
    for subkey in key
        haskey(subtrie.children, subkey) || return false
        subtrie = subtrie.children[subkey]
    end
    return !isnothing(subtrie.value)
end

function subtrie(trie, prefix)
    strie = trie
    for subkey in prefix
        haskey(strie.children, subkey) || return nothing
        strie = strie.children[subkey]
    end
    return strie
end
function subtrie!(trie, prefix)
    strie = trie
    for subkey in prefix
        strie = get!(() -> similar(trie), strie.children, subkey)
    end
    return strie
end

function Base.getindex(trie::Trie{K}, key::Vector{K}) where {K}
    strie = subtrie(trie, key)
    (isnothing(strie) || isnothing(strie.value)) && throw(KeyError("$key not in trie"))
    return strie.value
end

function Base.get(trie::Trie, key, default)
    strie = subtrie(trie, key)
    if isnothing(strie) || isnothing(strie.value)
        return default
    else
        return strie.value
    end
end
function Base.get(f::Base.Callable, trie::Trie, key)
    strie = subtrie(trie, key)
    if isnothing(strie) || isnothing(strie.value)
        return f()
    else
        return strie.value
    end
end

function Base.get!(trie::Trie{K, V}, key::Vector{K}, default::V) where {K, V}
    strie = subtrie!(trie, key)
    if isnothing(strie.value)
        strie.value = default
    end
    return strie.value
end
function Base.get!(trie::Trie{K}, key::Vector{K}, default) where {K}
    strie = subtrie!(trie, key)
    if isnothing(strie.value)
        strie.value = convert(valtype(trie), default)
    end
    return strie.value
end
function Base.get!(f::Base.Callable, trie::Trie{K}, key::Vector{K}) where {K}
    strie = subtrie!(trie, key)
    if isnothing(strie.value)
        strie.value = convert(valtype(trie), f())
    end
    return strie.value
end

function Base.setindex!(trie::Trie{K, V}, v::V, key::Vector{K}) where {K, V}
    strie = subtrie!(trie, key)
    strie.value = v
    return trie
end
function Base.setindex!(trie::Trie{K}, v, key::Vector{K}) where {K}
    strie = subtrie!(trie, key)
    strie.value = convert(valtype(trie), v)
    return trie
end

function Base.length(trie::Trie)
    n = 0
    for _ in trie
        n += 1
    end
    return n
end

Base.:(==)(trie1::Trie, trie2::Trie) = (trie1.value == trie2.value && trie1.children == trie2.children)
Base.isequal(trie1::Trie, trie2::Trie) = isequal(trie1.value, trie2.value) && isequal(trie1.children, trie2.children)

Base.isempty(trie::Trie) = isempty(trie.children) && isnothing(trie.value)

struct TrieKeyIterator{K, V}
    trie::Trie{K, V}
end
struct TriePairIterator{K, V}
    trie::Trie{K, V}
end

Base.keys(trie::Trie) = TrieKeyIterator(trie)
Base.pairs(trie::Trie) = TriePairIterator(trie)

Base.IteratorSize(::Type{<:TrieKeyIterator}) = Base.SizeUnknown()
Base.IteratorSize(::Type{<:TriePairIterator}) = Base.SizeUnknown()
Base.eltype(::Type{TrieKeyIterator{K, V}}) where {K, V} = Vector{K}
Base.eltype(::Type{TriePairIterator{K, V}}) where {K, V} = Pair{Vector{K}, V}

function Base.iterate(ki::TrieKeyIterator)
    next = iterate(ki.trie)
    isnothing(next) && return nothing
    _, state = next
    return copy(state[1]), state
end
function Base.iterate(ki::TrieKeyIterator, state)
    next = iterate(ki.trie, state)
    isnothing(next) && return nothing
    _, state = next
    return copy(state[1]), state
end

function Base.iterate(pi::TriePairIterator)
    next = iterate(pi.trie)
    isnothing(next) && return nothing
    val, state = next
    return copy(state[1]) => val, state
end
function Base.iterate(pi::TriePairIterator, state)
    next = iterate(pi.trie, state)
    isnothing(next) && return nothing
    val, state = next
    return copy(state[1]) => val, state
end

function Base.isassigned(trie::Trie{K}, key::Vector{K}) where {K}
    strie = subtrie(trie, key)
    return !isnothing(strie) && !isnothing(strie.value)
end

function Base.iterate(trie::Trie{K, V}) where {K, V}
    keystack = K[]
    nodestack = Trie{K, V}[trie]
    statestack = Int[]
    return _trie_iterate(keystack, nodestack, statestack)
end

function Base.iterate(::Trie, (keystack, nodestack, statestack))
    return _trie_iterate(keystack, nodestack, statestack)
end

function _trie_iterate(keystack, nodestack, statestack)
    while !isempty(nodestack)
        node = last(nodestack)

        if length(statestack) == length(keystack)
            # First visit to this node: emit value if present, then set up child iteration
            push!(statestack, 0)
            if !isnothing(node.value)
                return node.value, (keystack, nodestack, statestack)
            end
        end

        # Iterate children
        next = iterate(pairs(node.children), pop!(statestack))
        if !isnothing(next)
            (k, child), nstate = next
            push!(keystack, k)
            push!(nodestack, child)
            push!(statestack, nstate)
        else
            # Backtrack
            pop!(nodestack)
            isempty(keystack) && return nothing
            pop!(keystack)
        end
    end
    return nothing
end

# Utility
# -------
function Dictionaries.sortkeys!(trie::Trie)
    foreach(sortkeys!, trie.children)
    sortkeys!(trie.children)
    return trie
end
function Dictionaries.sortkeys(trie::Trie{K, V}) where {K, V}
    trie_sorted = Trie{K, V}(trie.value)
    for (k, v) in pairs(trie.children)
        insert!(trie_sorted.children, k, sortkeys(v))
    end
    sortkeys!(trie_sorted.children)
    return trie_sorted
end

# Convert
# -------

function Base.convert(::Type{Trie{K, V}}, trie::Trie) where {K, V}
    trie isa Trie{K, V} && return trie

    result = Trie{K, V}(trie.value)
    for (k, v) in pairs(trie.children)
        insert!(result.children, k, v)
    end
    return result
end


# Printing
# --------
function Base.show(io::IO, ::MIME"text/plain", trie::Trie)
    n = length(trie)
    print(io, typeof(trie), " (", n, n == 1 ? " entry" : " entries", ")")
    isempty(trie) && return
    # Root value corresponds to the empty key []
    has_root_val = !isnothing(trie.value)
    if has_root_val
        has_ch = !isempty(trie.children)
        print(io, '\n', has_ch ? "├─ " : "└─ ", "[] => ")
        show(io, trie.value)
        has_ch || return
    end
    _trie_show_children(io, trie.children, Bool[])
    return nothing
end

# Print the children of a trie node. Each entry is preceded by a newline so
# that the output never ends with a trailing newline (Julia show convention).
# ancestors_last[i] is true if ancestor i was the last child of its parent
# (used to decide whether to print "│  " or "   " for the indent).
function _trie_show_children(io::IO, children, ancestors_last::Vector{Bool})
    n = length(children)
    for (i, (k, node)) in enumerate(Dictionaries.pairs(children))
        is_last = i == n
        print(io, '\n')
        for anc_last in ancestors_last
            print(io, anc_last ? "   " : "│  ")
        end
        print(io, is_last ? "└─ " : "├─ ")
        # Path compression: follow single-child, no-value chains into one edge label
        edge = [k]
        cur = node
        while isnothing(cur.value) && length(cur.children) == 1
            next_k, next_node = only(Dictionaries.pairs(cur.children))
            push!(edge, next_k)
            cur = next_node
        end
        _trie_show_edge(io, edge)
        if !isnothing(cur.value)
            print(io, " => ")
            show(io, cur.value)
        end
        isempty(cur.children) || _trie_show_children(io, cur.children, [ancestors_last; is_last])
    end
    return nothing
end

# Print an edge label: Char sequences always as strings, other sequences as
# a single element or a vector.
function _trie_show_edge(io::IO, edge::Vector{K}) where {K}
    if K === Char
        print(io, '"', join(edge), '"')
    elseif length(edge) == 1
        show(io, edge[1])
    else
        show(io, edge)
    end
end
