@doc """
    GlobalOp{T,A,S}

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
