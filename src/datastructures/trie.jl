# equal word length trie
mutable struct Trie{K} <: AbstractIndices{K}
    const children::Dictionary{K,Trie{K}}
    value::K

    function Trie{K}(value::K) where {K}
        children = Dictionary{K,Trie{K}}()
        return new{K}(children, value)
    end
end

# Constructors
# ------------
Trie() = Trie{Any}()
function Trie(ks)
    K = eltype(eltype(ks)) # K is the type of iterating the key
    return Trie{K}(ks)
end
function Trie{K}(ks) where {K}
    trie = Trie{K}(begin_marker(V))
    for (k, v) in zip(ks, vs)
        trie[k] = v
    end
    return trie
end

Base.similar(t::Trie) = Trie{keytype(t),valtype(t)}()

function Trie(vertices, ex::GlobalOp)
    A = algebratype(ex)
    root = Trie{A}(begin_marker(A))

    opstrings = operatorstrings(vertices, ex)

    for op in opstrings
        trie = root
        for o in op
            child = get!(trie.children, o) do
                Trie{keytype(trie)}(o)
            end
            trie = child
        end
    end

    return root
end

# Properties
# ----------
# Base.keytype(trie::Trie) = keytype(typeof(trie))
# Base.keytype(::Type{<:Trie{K}}) where {K} = K
# Base.valtype(trie::Trie) = valtype(typeof(trie))
# Base.valtype(::Type{<:Trie{K}}) where {K,V} = V

AbstractTrees.children(trie::Trie) = keys(trie.children)
AbstractTrees.nodevalue(trie::Trie) = trie.value

function depth(trie::Trie)
    d = 0
    strie = trie
    while !isempty(strie.children)
        d += 1
        strie = first(strie.children)
    end
    return d
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

function Base.getindex(trie::Trie, key)
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

function Base.get!(trie::Trie, key, default)
    strie = subtrie!(trie, key)
    if isnothing(strie.value)
        strie.value = convert(valtype(trie), default)
    end
    return strie.value
end
function Base.get!(f::Base.Callable, trie::Trie, key)
    strie = subtrie!(trie, key)
    if isnothing(strie.value)
        strie.value = convert(valtype(trie), f())
    end
    return strie.value
end

function Base.setindex!(trie::Trie, v, key)
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

# TODO: iterator
function Base.keys(trie::Trie)
    found = Vector{Vector{keytype(trie)}}(undef, 0)
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
    found = Vector{Pair{Vector{keytype(trie)},valtype(trie)}}(undef, 0)
    next = iterate(trie)
    while !isnothing(next)
        val, (keystack, statestack) = next
        push!(found, copy(keystack) => val) # copy required because memory reused!
        next = iterate(trie, (keystack, statestack))
    end
    return found
end

# Iterators
# ---------
Base.IteratorSize(::Type{<:Trie}) = Base.SizeUnknown()
Base.eltype(::Type{T}) where {T<:Trie} = valtype(T)

function Base.iterate(trie::Trie)
    keystack = keytype(trie)[]
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
function Base.sort!(trie::Trie)
    foreach(sort!, trie.children)
    sortkeys!(trie.children)
    return trie
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
