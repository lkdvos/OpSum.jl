struct DawgDictionary{I, T, C} <: AbstractDictionary{I, T}
    indices::DawgIndices{C, I}
    values::Vector{T}
end

Base.keys(dict::DawgDictionary) = getfield(dict, :indices)
_values(dict::DawgDictionary) = getfield(dict, :values)

# Constructors
# ------------

function DawgDictionary(words, values)
    K = eltype(eltype(words))
    V = eltype(values)
    trie = Trie{K, V}()
    for (w, v) in zip(words, values)
        trie[w] = v
    end
    return DawgDictionary(trie)
end

function DawgDictionary(trie::Trie)
    trie_sorted = sortkeys(trie)
    sorted_pairs = pairs(trie_sorted)
    sorted_words = first.(sorted_pairs)
    sorted_values = last.(sorted_pairs)
    inds = DawgIndices(sorted_words)
    return DawgDictionary(inds, sorted_values)
end

depth(dict::DawgDictionary) = depth(keys(dict))
state_registers(dict::DawgDictionary, d) = state_registers(keys(dict), d)
Base.isempty(dict::DawgDictionary) = isempty(dict.values)
Base.length(dict::DawgDictionary) = length(dict.values)

# Tokens
# ------
Dictionaries.istokenizable(dict::DawgDictionary) = true

Dictionaries.tokenized(dict::DawgDictionary) = _values(dict)

function Dictionaries.istokenassigned(dict::DawgDictionary, (_slot, index))
    return isassigned(_values(dict), index)
end
function Dictionaries.istokenassigned(dict::DawgDictionary, index::Int)
    return isassigned(_values(dict), index)
end

@inline Dictionaries.gettokenvalue(dict::DawgDictionary, (_slot, index)) =
    _values(dict)[index]
@inline Dictionaries.gettokenvalue(dict::DawgDictionary, index::Int) = _values(dict)[index]

# Prefix/Suffix queries
# ---------------------

function _prefix_token_start(inds::DawgIndices, prefix)
    token = 0
    state = root(inds)
    for subkey in prefix
        for (k, child) in pairs(state.children)
            if k == subkey
                state = child
                break
            else
                token += length(child)
            end
        end
    end
    return token
end

function _collect_suffixes!(results, values, node, suffix, token_offset, remaining_depth)
    if remaining_depth == 0
        token = token_offset + length(results) + 1
        push!(results, copy(suffix) => values[token])
        return
    end
    for (k, child) in pairs(node.children)
        push!(suffix, k)
        _collect_suffixes!(results, values, child, suffix, token_offset, remaining_depth - 1)
        pop!(suffix)
    end
end

"""
    suffixes_with_values(dict::DawgDictionary, prefix)

Return all `suffix => value` pairs for words in `dict` that start with `prefix`.
The suffix is the portion of the word after `prefix`.
Returns an empty vector if `prefix` is not found.
"""
function suffixes_with_values(dict::DawgDictionary{I, T, C}, prefix) where {I, T, C}
    inds = keys(dict)
    node = subdawg(root(inds), prefix)
    isnothing(node) && return Pair{Vector{C}, T}[]
    token_start = _prefix_token_start(inds, prefix)
    suffix_depth = depth(inds) - length(prefix)
    results = Pair{Vector{C}, T}[]
    _collect_suffixes!(results, _values(dict), node, Vector{C}(), token_start, suffix_depth)
    return results
end

"""
    prefixes_at_depth(dict::DawgDictionary, d)

Return all unique prefixes of length `d` present in `dict`.
"""
prefixes_at_depth(dict::DawgDictionary, d) = prefixes_at_depth(keys(dict), d)

const DAWGDictionary = DawgDictionary
