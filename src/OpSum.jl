module OpSum

export Trie
export SDAWG
export DAWGDictionary

using Dictionaries

# Data structures
# ---------------
include("trie.jl")
include("dawg.jl")
include("dawgdict.jl")

end
