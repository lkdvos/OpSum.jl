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
    trie = Trie{K, V}(begin_marker(V))
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


# TODO: iterator
function Base.keys(trie::Trie)
    found = keytype(trie)[]
    next = iterate(trie)
    while !isnothing(next)
        _, (keystack, statestack) = next
        push!(found, copy(keystack)) # copy required because memory reused!
        next = iterate(trie, (keystack, statestack))
    end
    return found
end

# TODO: iterator
function Base.pairs(trie::Trie)
    found = Vector{Pair{keytype(trie), valtype(trie)}}(undef, 0)
    next = iterate(trie)
    while !isnothing(next)
        val, (keystack, statestack) = next
        push!(found, copy(keystack) => val) # copy required because memory reused!
        next = iterate(trie, (keystack, statestack))
    end
    return found
end

function Base.isassigned(trie::Trie{K}, key::Vector{K}) where {K}
    strie = subtrie(trie, key)
    return !isnothing(strie) && !isnothing(strie.value)
end

# Iterators
# ---------
Base.IteratorSize(trie::Trie) = Base.IteratorSize(typeof(trie))
Base.IteratorSize(::Type{T}) where {T <: Trie} = Base.SizeUnknown()
Base.eltype(::Type{T}) where {T <: Trie} = valtype(T)

function Base.iterate(trie::Trie)
    keystack = keytype(trie)()
    statestack = Int[0]
    return iterate(trie, (keystack, statestack))
end

function Base.iterate(trie::Trie, (keystack, statestack))
    strie = subtrie(trie, keystack)

    # check if need to go deeper
    if length(statestack) == length(keystack)
        push!(statestack, 0)
        if isnothing(strie.value) || !isempty(strie.children)
            return iterate(trie, (keystack, statestack))
        else
            return strie.value, (keystack, statestack)
        end
    end

    # check at current level
    next = iterate(pairs(strie.children), pop!(statestack))
    if !isnothing(next)
        (k, child), nstate = next
        push!(keystack, k)
        push!(statestack, nstate)
        return iterate(trie, (keystack, statestack))
    end

    # check if need to backtrack
    isempty(keystack) && return nothing
    pop!(keystack)
    return iterate(trie, (keystack, statestack))
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
# function Base.show(io::IO, ::MIME"text/plain", trie::Trie)
#     show(io, typeof(trie))
#     println(io, ":")
#     iob = IOContext(io, :typeinfo => keytype(trie))
#     return show_trie(iob, trie)
# end

show_trie(io, trie::Trie) = AbstractTrees.print_tree(io, trie)

# function show_trie(io, trie, prefix=keytype(trie)[])
#     if !isnothing(trie.value)
#         println(io, prefix, " => ", trie.value)
#     end
#     for (k, v) in pairs(trie.children)
#         show_trie(io, v, vcat(prefix, k))
#     end
#     return nothing
# end
