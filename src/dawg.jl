# TODO: Sorteddict?
using Dictionaries

"""
    SDAWG{K,I} <: AbstractIndices{I}

Static Directed Acyclic Word Graph for words of type `W` with characters of type `K`.

`SDAWG` make efficient indices from sorted input lists of words, with queries scaling
linearly with word length. Importantly, they provide compressed storage schemes by
bundling common prefixes and suffixes.

Importantly, here we assume that all words are of equal lenght, which saves slightly on
storage space as well as efficiency.
"""
mutable struct SDAWG{K,I} <: AbstractIndices{I}
    children::Dictionary{K,SDAWG{K}}
    descendants::Int
    function SDAWG{K,I}() where {K,I}
        children = Dictionary{K,SDAWG{K}}()
        return new{K,I}(children, 0)
    end
end

function SDAWG(list::AbstractVector{I}) where {I}
    sortedlist = unique!(sort(list))
    dawg = SDAWG{eltype(eltype(sortedlist)),I}()

    register = Vector{typeof(dawg)}(undef, 0)
    # push!(register, similar(dawg))
    for word in sortedlist
        @info "adding word:" word
        prefix = common_prefix(dawg, word)
        state = subdawg(dawg, prefix)
        suffix = @view(word[(length(prefix) + 1):end])

        replace_or_register!(register, state)
        @info "length register:" register
        add_suffix!(state, suffix)
    end
    replace_or_register!(register, dawg)

    return dawg
end

Base.length(dawg::SDAWG) = max(1, sum(length, dawg.children; init=0))

Base.similar(::SDAWG{K,I}) where {K,I} = SDAWG{K,I}()

# This is an optimized version of == which may be used whenever a
# dawg is being constructed since we know traversal will be going on
# in lexicographic order: two states are equal iff
# 1. both are final or both are nonfinal
# 2. they have the same number of outgoing transitions
# 3. outgoing transitions have the same labels
# 4. corresponding transitions lead to equivalent states
# Here, because of the order, we have instead of 4.
# 4′. corresponding transitions lead to the same states.
function issameclass(dawg1::SDAWG, dawg2::SDAWG)
    # return false
    length(dawg1) == length(dawg2) || return false
    isdictequal(keys(dawg1.children), keys(dawg2.children)) || return false
    return all(Base.Splat(===), zip((dawg1.children), (dawg2.children)))
end
issameclass(dawg::SDAWG) = Base.Fix1(issameclass, dawg)

function subdawg(dawg::SDAWG, prefix)
    sdawg = dawg
    for subkey in prefix
        haskey(sdawg.children, subkey) || return nothing
        sdawg = sdawg.children[subkey]
    end
    return sdawg
end

function common_prefix(dawg::SDAWG, word)
    sdawg = dawg
    for (i, subkey) in enumerate(word)
        haskey(sdawg.children, subkey) || return @view(word[1:(i - 1)])
        sdawg = sdawg.children[subkey]
    end
    @assert false
    return @view(word[1:end])
end

function replace_or_register!(register, state)
    isempty(state.children) && return register
    child = last(state.children)
    replace_or_register!(register, child)

    i = findfirst(issameclass(child), register)
    if !isnothing(i)
        # @info "found" register[i] child
        state.children[end] = register[i]
        state.descendants = max(1, length(state) + length(register[i]))
    else
        push!(register, child)
        # @info "added" child register
    end
    return register
end

function add_suffix!(dawg::SDAWG, suffix)
    # dawg.descendants += 1
    for c in suffix
        child = similar(dawg)
        insert!(dawg.children, c, child)
        @assert issorted(keys(dawg.children))
        dawg.descendants += 1
        dawg = child
    end
    return dawg
end

function Base.iterate(dawg::SDAWG{K}, i=1) where {K}
    i > length(dawg) && return nothing
    return gettokenvalue(dawg, i), i + 1
end

# function Base.iterate(indices::Iterators.Reverse{<:SDAWG}, i=length(indices.itr))
#     i < 1 && return nothing
#     return gettokenvalue(indices.itr, i), i - 1
# end

# Token interface
# ---------------
Dictionaries.istokenizable(::SDAWG) = true
Dictionaries.tokentype(::SDAWG) = Int

Dictionaries.iteratetoken(indices::SDAWG, state...) = iterate(1:length(indices), state...)
function Dictionaries.iteratetoken_reverse(indices::SDAWG, state...)
    return iterate(Iterators.reverse(1:length(indices)), state...)
end

function Dictionaries.gettoken(indices::SDAWG, key)
    token = 0
    state = indices
    for subkey in key
        for k in keys(state.children)
            if k == subkey
                state = state.children[k]
                break
            else
                token += length(child)
            end
        end
        if isempty(state.children)
            token += 1
        else
            return false, token
        end
    end
    return true, token
end

function Dictionaries.gettokenvalue(indices::SDAWG{K}, token) where {K}
    @boundscheck 0 ≤ token ≤ length(indices) || throw(KeyError(token))
    # @info "getting token" token
    state = indices
    keystack = Vector{K}(undef, 0)
    while token > 0
        # @show token state.children length(state)
        if isempty(state.children)
            token -= 1
            break
        end
        for (k, child) in pairs(state.children)
            if token <= max(1, length(child))
                push!(keystack, k)
                state = child
                break
            else
                token -= max(1, length(child))
            end
        end
        # @assert false
    end

    return _create_key(indices, keystack)
end

_create_key(::SDAWG{K,I}, keystack::I) where {K,I} = keystack
_create_key(::SDAWG{Char,String}, keystack::Vector{Char}) = join(keystack)
_create_key(::SDAWG{K,Tuple{Vararg{K}}}, keystack::Vector{K}) where {K} = Tuple(keystack)

# Printing
# --------
# function Base.show(io::IO, ::MIME"text/plain", dawg::SDAWG)
#     show(io, typeof(dawg))
#     println(io, "of length: ", dawg.descendants)
#     if isempty(dawg.children)
#         println(io, " empty")
#     else
#         show_dawg(io, dawg)
#     end
#     return nothing
# end

function show_dawg(io, dawg, prefix="  ")
    println(io, dawg.descendants)
    if isempty(dawg.children)
        println(io)
        print(io, prefix)
        return nothing
    end
    for (k, v) in pairs(dawg.children)
        show_dawg(io, v, prefix * string(k))
    end
    return nothing
end

function _num_unique_nodes(indices::SDAWG)
    s = Set(objectid(indices))
    for c in indices.children
        _add_nodes!(s, c)
    end
    return length(s)
end
function _add_nodes!(s::Set, indices::SDAWG)
    push!(s, objectid(indices))
    for c in indices.children
        _add_nodes!(s, c)
    end
end
