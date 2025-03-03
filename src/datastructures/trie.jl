# equal word length trie
mutable struct Trie{K,V}
    const children::Dict{K,Trie{K,V}}
    value::Union{Nothing,V}

    function Trie{K,V}() where {K,V}
        children = Dict{K,Trie{K,V}}()
        return new{K,V}(children, nothing)
    end
end

# Constructors
# ------------
Trie() = Trie{Any,Any}()
function Trie(ks, vs)
    K = eltype(eltype(ks)) # K is the type of iterating the key
    V = eltype(vs)
    return Trie{K,V}(ks, vs)
end
function Trie{K,V}(ks, vs) where {K,V}
    trie = Trie{K,V}()
    for (k, v) in zip(ks, vs)
        trie[k] = v
    end
    return trie
end

Base.similar(t::Trie) = Trie{keytype(t),valtype(t)}()

# Properties
# ----------
Base.keytype(trie::Trie) = keytype(typeof(trie))
Base.keytype(::Type{<:Trie{K}}) where {K} = K
Base.valtype(trie::Trie) = valtype(typeof(trie))
Base.valtype(::Type{<:Trie{K,V}}) where {K,V} = V

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

function Base.iterate(
    trie::Trie,
    (keystack, statestack)=(Vector{keytype(trie)}(undef, 0), [trie.children.idxfloor]),
)
    strie = subtrie(trie, keystack)
    next = iterate(strie.children, statestack[end])

    if isnothing(next)
        isempty(keystack) && return nothing
        pop!(keystack)
        pop!(statestack)
        return iterate(trie, (keystack, statestack))
    end

    (k, child), statestack[end] = next
    push!(keystack, k)
    push!(statestack, child.children.idxfloor)

    if !isnothing(child.value)
        return child.value, (keystack, statestack)
    else
        return iterate(trie, (keystack, statestack))
    end
end

# Printing
# --------
function Base.show(io::IO, ::MIME"text/plain", trie::Trie)
    show(io, typeof(trie))
    println(io, ":")
    return show_trie(io, trie)
end

function show_trie(io, trie, prefix="  ")
    if !isnothing(trie.value)
        println(io, prefix, " => ", trie.value)
    end
    for (k, v) in trie.children
        show_trie(io, v, prefix * string(k))
    end
    return nothing
end
