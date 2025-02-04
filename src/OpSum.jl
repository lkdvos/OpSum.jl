module OpSum

export Trie
export SDAWG
export DAWGDictionary
export opsum_to_fsm

using Dictionaries

# Data structures
# ---------------
include("trie.jl")
include("dawg.jl")
include("dawgdict.jl")

# Operators
# ---------
include("operators.jl")
include("state_machines.jl")
include("compression.jl")

end
