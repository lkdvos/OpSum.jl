struct DAWGDictionary{I,T,C} <: AbstractDictionary{I,T}
    indices::SDAWGIndices{C,I}
    values::Vector{T}
end

Base.keys(dict::DAWGDictionary) = getfield(dict, :indices)
_values(dict::DAWGDictionary) = getfield(dict, :values)

# Constructors
# ------------

function DAWGDictionary(inds, values)
    I = sortperm(inds)
    indices_dawg = SDAWGIndices(inds[I])
    return DAWGDictionary(indices_dawg, values[I])
end

depth(dict::DAWGDictionary) = depth(keys(dict))
state_register(dict::DAWGDictionary, d) = state_register(keys(dict), d)
Base.isempty(dict::DAWGDictionary) = isempty(dict.values)
Base.length(dict::DAWGDictionary) = length(dict.values)

# Tokens
# ------
Dictionaries.istokenizable(dict::DAWGDictionary) = true

Dictionaries.tokenized(dict::DAWGDictionary) = _values(dict)

function Dictionaries.istokenassigned(dict::DAWGDictionary, (_slot, index))
    return isassigned(_values(dict), index)
end
function Dictionaries.istokenassigned(dict::DAWGDictionary, index::Int)
    return isassigned(_values(dict), index)
end

@inline Dictionaries.gettokenvalue(dict::DAWGDictionary, (_slot, index)) =
    _values(dict)[index]
@inline Dictionaries.gettokenvalue(dict::DAWGDictionary, index::Int) = _values(dict)[index]
