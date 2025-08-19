
"""
    DawgNode{K}

Static Directed Acyclic Word Graph for words of type `W` with characters of type `K`.

`DawgNode` make efficient indices from sorted input lists of words, with queries scaling
linearly with word length. Importantly, they provide compressed storage schemes by
bundling common prefixes and suffixes.

Importantly, here we assume that all words are of equal lenght, which saves slightly on
storage space as well as efficiency.
"""
mutable struct DawgNode{K}
    const children::Dictionary{K,DawgNode{K}}
    descendants::Int

    function DawgNode{K}() where {K}
        children = Dictionary{K,DawgNode{K}}()
        return new{K}(children, 0)
    end
end

#=
Interesting note: I didn't figure out how to make the dictionary type parametric, and I am
thinking it is not possible with the Julia type system.
The naive approach seems to lead to some infinite recursion problems:
struct DawgNode{K,D}
    children::D
    descendants::Int
    function DawgNode{K,D}() where {K,D}
        # here D == Dictionary{K,DawgNode{K,Dictionary{K,DawgNode{K,...}}}}
        children = D()
        return new{K,D}(children, 0)
    end
end
=#

Base.similar(::DawgNode{K}) where {K} = DawgNode{K}()

# Properties
# ----------
AbstractTrees.children(inds::DawgNode) = inds.children
AbstractTrees.nodevalue(inds::DawgNode) = objectid(inds) # inds.descendants
AbstractTrees.childtype(dawg::DawgNode) = typeof(dawg)
Base.length(dawg::DawgNode) = max(1, sum(length, dawg.children; init=0))

Base.keytype(dawg::DawgNode) = keytype(dawg.children)
# Base.length(dawg::DawgNode) = dawg.descendants

function depth(dawg::DawgNode)
    d = 0
    state = dawg
    while !isempty(state.children)
        state = first(state.children)
        d += 1
    end
    return d
end

# TODO: remove
function DawgNode(list)
    sortedlist = unique!(sort(list))
    root = DawgNode{eltype(eltype(sortedlist))}()
    root, _ = _sdawg!(root, sortedlist)
    return root
end

function _sdawg!(root, sortedlist)
    dawg = root
    register = Vector{typeof(dawg)}(undef, 0)
    # push!(register, similar(dawg))
    for word in sortedlist
        prefix = common_prefix(dawg, word)
        state = subdawg(dawg, prefix)
        suffix = @view(word[(length(prefix) + 1):end])

        replace_or_register!(register, state)
        add_suffix!(state, suffix)
    end
    replace_or_register!(register, dawg)

    return dawg, register
end

# This is an optimized version of == which may be used whenever a
# dawg is being constructed since we know traversal is lexicographic:
# two states are equivalent if:
# 1. both are final or both are nonfinal
# 2. they have the same number of outgoing transitions
# 3. outgoing transitions have the same labels
# 4. corresponding transitions lead to equivalent states
# Here, because of the order, we have instead of 4.
# 4′. corresponding transitions lead to the same states.
function issameclass(dawg1::DawgNode, dawg2::DawgNode)
    # @assert depth(dawg1) == depth(dawg2) "depths must be equal"
    # length(dawg1) == length(dawg2) || return false
    isdictequal(keys(dawg1.children), keys(dawg2.children)) || return false
    return all(Base.Splat(===), zip((dawg1.children), (dawg2.children)))
end
issameclass(dawg::DawgNode) = Base.Fix1(issameclass, dawg)

function subdawg(dawg::DawgNode, prefix)
    sdawg = dawg
    for subkey in prefix
        haskey(sdawg.children, subkey) || return nothing
        sdawg = sdawg.children[subkey]
    end
    return sdawg
end

function partial_getindex(inds::DawgNode, prefix)
    state = subdawg(inds, prefix)
    isnothing(state) && throw(BoundsError(inds, prefix))
    return state
end

function common_prefix(dawg::DawgNode, word)
    sdawg = dawg
    for (i, subkey) in enumerate(word)
        haskey(sdawg.children, subkey) || return @view(word[1:(i - 1)])
        sdawg = sdawg.children[subkey]
    end
    @assert false
    return @view(word[1:end])
end

# TODO: tail-recursion to simple loop ?
"""
    replace_or_register!(register, state::DawgNode)

Main loop for minimizing the word graph.
Given a register of previously seen states, merge `state` into it.
"""
function replace_or_register!(registers, state::DawgNode)
    if isempty(state.children)
        return registers
        length(registers) < 1 || @warn "wrong" length(registers) state
        @assert length(registers) < 1 "should be empty"
        return registers
    end
    # depth first search for minimizing
    child = last(state.children)
    replace_or_register!(@view(registers[2:end]), child)

    register = first(registers)
    i = findfirst(issameclass(child), register)
    if !isnothing(i) # state already exists, so merge
        # state.descendants = length(state) + length(register[i])
        # state.descendants = max(1, length(state) + 0length(register[i]))
        # @info "found" i register child
        state.children[end] = register[i]
        state.descendants = length(state)
    else # state is new, register it
        push!(register, child)
        # @error "new term:" register
    end

    return registers
end

function add_suffix!(dawg::DawgNode, suffix)
    for c in suffix
        child = similar(dawg)
        child.descendants = 1
        insert!(dawg, c, child)
        dawg.descendants = length(dawg)
        @assert issorted(keys(dawg.children)) "should be constructed in lexicographic order"
        dawg = child
    end
    return dawg
end

function Dictionaries.insert!(dawg::DawgNode, key, value)
    insert!(children(dawg), key, value)
    # dawg.descendants += length(value)
    return dawg
end

# Base.show(io::IO, dawg::DawgNode) = AbstractTrees.print_tree(io, dawg)

function Base.iterate(dawg::DawgNode, iterstate=1)
    token = iterstate
    0 ≤ token ≤ length(dawg) || return nothing
    state = dawg
    keystack = empty_prefix(dawg)
    while token > 0
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
    end

    return keystack, iterstate + 1
end
# utility
# _create_key(::DawgIndices{K,I}, keystack::I) where {K,I} = keystack
# _create_key(::DawgIndices{Char,String}, keystack::Vector{Char}) = join(keystack)
# function _create_key(::DawgIndices{K,Tuple{Vararg{K}}}, keystack::Vector{K}) where {K}
#     return Tuple(keystack)
# end

empty_prefix(node::DawgNode) = Vector{keytype(node)}(undef, 0)

# ------------------------------------------------------------------------------------------------------
# store registers to improve efficiency
"""
    DawgIndices{K,W} <: AbstractIndices{W}

Directed Acyclic Word Graph for words of type `W` with characters of type `K`.
"""
struct DawgIndices{K,W} <: AbstractIndices{W}
    registers::Vector{Vector{DawgNode{K}}}
end

function DawgIndices(vertices, allterms)
    lettertype = eltype(eltype(allterms))

    # initialize starting and ending states
    nodetype = DawgNode{lettertype}
    # root.descendants = 1
    registers = map(x -> nodetype[], 1:(length(vertices) + 1))
    root = nodetype()
    root.descendants = 1

    # add bookkeeping:
    terms = collect(allterms)
    @assert issorted(terms) "terms must be sorted"
    ops_bookkeeping = map(0:length(vertices)) do i
        return mapreduce(*, 1:length(vertices)) do j
            Operator(j <= i ? :B : :E, j)
        end
    end
    append!(terms, ops_bookkeeping)
    allterms = allterms + sum(ops_bookkeeping)
    # sort!(terms)

    @debug "building dawg" sortedterms(allterms) registers
    stest = sort!(map(x -> operatorstring(vertices, x), sortedterms(allterms)))

    # @info "checking order" stest sortperm(stest)
    for word in stest
        state = root
        state.descendants += 1
        # word = operatorstring(vertices, term)

        # search for common prefix
        @show prefix = common_prefix(root, word)
        @show state = subdawg(root, prefix)
        @show suffix = @view(word[(length(prefix) + 1):end])

        # minimize word graph
        replace_or_register!(registers[(length(prefix) + 2):end], state)
        add_suffix!(state, suffix)

        @debug "added term" term root
    end
    replace_or_register!(registers[2:end], root)

    @debug "built dawg" registers

    push!(registers[1], root)
    return DawgIndices{lettertype,Vector{lettertype}}(registers)
end

# DFS traversal of the trie to find common suffixes
function DawgIndices(trie::Trie{K}) where {K}
    # @assert issorted(trie) "trie must be sorted"
    root = DawgNode{K}()
    root.descendants = 1
    registers = map(x -> typeof(root)[], 1:depth(trie))
    # push!(registers, root)

    _dawgindices!(root, trie, @view(registers[1:end]))
    pushfirst!(registers, [root])

    return DawgIndices{K,Vector{K}}(registers)
end
function _dawgindices!(root, trie, registers)
    for (op, child) in pairs(trie.children)
        # depth first search
        child_dawg = similar(root)
        child_dawg.descendants = 1
        _dawgindices!(child_dawg, child, @view(registers[2:end]))

        # minimize
        i = findfirst(issameclass(child_dawg), first(registers))
        if isnothing(i)
            if isbegin(op)
                pushfirst!(first(registers), child_dawg) # starting state should be first
            elseif isend(op)
                @assert isbegin(first(registers)[1]) "begin should be first"
                insert!(first(registers), 2, child_dawg) # starting state should already be there?
            else
                push!(first(registers), child_dawg)
            end
        else
            child_dawg = first(registers)[i]
        end

        insert!(root, op, child_dawg)
    end
end

root(inds::DawgIndices) = only(state_registers(inds, 0))
state_registers(inds::DawgIndices) = inds.registers
state_registers(inds::DawgIndices, d::Int) = inds.registers[d + 1]

Base.length(inds::DawgIndices) = length(root(inds))
depth(inds::DawgIndices) = depth(root(inds))

partial_getindex(inds::DawgIndices, prefix) = partial_getindex(root(inds), prefix)

AbstractTrees.children(inds::DawgIndices) = children(root(inds))

# Token interaface
# ----------------
# Dictionaries.istokenizable(::SDAWGIndices) = true
# Dictionaries.tokentype(::SDAWGIndices) = Int

# function Dictionaries.iteratetoken(indices::SDAWGIndices, state...)
#     return iterate(1:length(indices), state...)
# end
# function Dictionaries.iteratetoken_reverse(indices::SDAWGIndices, state...)
#     return iterate(Iterators.reverse(1:length(indices)), state...)
# end

# Dictionaries.gettoken(inds::SDAWGIndices, key) = gettoken(root(inds), key)
# Dictionaries.gettokenvalue(inds::SDAWGIndices, token) = gettokenvalue(root(inds), token)

# Token interface
# ---------------
Dictionaries.istokenizable(::DawgIndices) = true
Dictionaries.tokentype(::DawgIndices) = Int

function Dictionaries.iteratetoken(indices::DawgIndices, state...)
    return iterate(1:length(indices), state...)
end
function Dictionaries.iteratetoken_reverse(indices::DawgIndices, state...)
    return iterate(Iterators.reverse(1:length(indices)), state...)
end

function Dictionaries.gettoken(indices::DawgIndices, key)
    token = 0
    state = root(indices)
    for (d, subkey) in enumerate(key)
        if d != length(key) && isempty(state.children)
            return false, token
        end
        for k in keys(state.children)
            if k == subkey
                state = state.children[k]
                break
            else
                token += length(state.children[k])
            end
        end
    end
    return true, token + 1
end

function Dictionaries.gettokenvalue(indices::DawgIndices{K}, token) where {K}
    @boundscheck 0 ≤ token ≤ length(indices) || throw(KeyError(token))
    # @info "getting token" token
    # token += 1
    state = root(indices)
    keystack = empty_prefix(indices)
    while token > 0
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
    end

    return _create_key(indices, keystack)
end

# utility
_create_key(::DawgIndices{K,I}, keystack::I) where {K,I} = keystack
_create_key(::DawgIndices{Char,String}, keystack::Vector{Char}) = join(keystack)
function _create_key(::DawgIndices{K,Tuple{Vararg{K}}}, keystack::Vector{K}) where {K}
    return Tuple(keystack)
end

empty_prefix(inds::DawgIndices) = Vector{eltype(keytype(inds))}(undef, 0)

function _num_unique_nodes(indices::DawgIndices)
    s = Set(objectid(indices))
    for c in indices.children
        _add_nodes!(s, c)
    end
    return length(s)
end
function _add_nodes!(s::Set, indices::DawgIndices)
    push!(s, objectid(indices))
    for c in indices.children
        _add_nodes!(s, c)
    end
end
