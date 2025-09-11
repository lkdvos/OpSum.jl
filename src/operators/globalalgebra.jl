struct SiteOp{T, A, S}
    op::LocalOp{T, A}
    sites::Vector{S}
end

@doc """
    GlobalOp{T,A,S} <: SymbolicAlgebra{T}

Symbolic operator algebra for representing operators acting on (a subset of) a global Hilbert space.
The type perameter `T` is the scalar type, `A` denotes the algebra type, and `S` is the type of the space.
""" GlobalOp

@sumtype GlobalOp{T, A, S}(
    SiteOp{T, A, S},
    #   Scaled{T,GlobalOp{T,A,S}},
    Sum{T, GlobalOp{T, A, S}},
    Prod{GlobalOp{T, A, S}},
    Pow{GlobalOp{T, A, S}},
    Fun{GlobalOp{T, A, S}},
) <: SymbolicAlgebra{T}

# Syntax: O[inds...] means O applied to inds
function Base.getindex(O::LocalOp, inds::S...) where {S}
    indices = S[inds...]
    T = scalartype(O)
    A = algebratype(O)
    return GlobalOp{T, A, S}(SiteOp{T, A, S}(O, indices))
end

# Incorporating lattice information
# ---------------------------------

function Trie(vertices, ex::GlobalOp)
    A = algebratype(ex)
    V = scalartype(ex)
    root = Trie{A, V}()

    coefficients, opstrings = operatorstrings(vertices, ex)

    for (c, op) in zip(coefficients, opstrings)
        trie = root
        for o in op
            child = get!(trie.children, o) do
                Trie{A, V}()
            end
            trie = child
        end
        @assert isnothing(trie.value) "Duplicate values?"
        trie.value = c
    end

    return sortkeys!(root)
end
