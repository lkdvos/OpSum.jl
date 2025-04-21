struct DawgDictionary{I,T,C} <: AbstractDictionary{I,T}
    indices::DawgIndices{C,I}
    values::Vector{T}
end

Base.keys(dict::DawgDictionary) = getfield(dict, :indices)
_values(dict::DawgDictionary) = getfield(dict, :values)

# Constructors
# ------------

function DawgDictionary(inds, values)
    I = sortperm(inds)
    indices_dawg = DawgIndices(inds[I])
    return DawgDictionary(indices_dawg, values[I])
end

depth(dict::DawgDictionary) = depth(keys(dict))
state_register(dict::DawgDictionary, d) = state_register(keys(dict), d)
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
