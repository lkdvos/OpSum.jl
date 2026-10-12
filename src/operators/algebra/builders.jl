# Hermitian conjugation of an `OperatorSum` (`H'`). The named operator sets live in stockbases.jl.

using TensorKit: ElementarySpace, Sector, sectortype, dual, removeunit

# Hermitian conjugation
# =====================
# `H'` is what turns a hopping amplitude into a hermitian Hamiltonian, and there is no cheap symbolic
# route to it. The adjoint of a term factorises over sites — `(f ⊗ g)† = f† ⊗ g†` in any dagger
# category, so no leg reversal is involved — but the *letter* adjoints do not stay in the alphabet
# position-by-position: `instantiate(letter, V)'` carries the dual charge and is generally a
# combination of that charge's letters, and re-expressing the caterpillar coupler over the dual
# charges is categorical data (B-symbols), not bookkeeping.
#
# So this goes the reliable way instead: materialise the term's `K`-site block through the tested
# forward map, take the adjoint there, and `project` it back. That is correct by construction —
# `project` re-materialises its own output and throws if it is not faithful — and the cost is one
# projection per *distinct* `(keys, spaces)`, memoised, rather than one per term.

const _ADJOINT_CACHE = Dict{Any, Any}()

# A `K`-site block back in the letters of its sites, each slot in its site's basis. The result leaves
# the span when a basis is not closed under adjoints, and then `project` throws with a hint to extend it.
_project_local(h, Bs::AbstractVector{<:Basis{I}}) where {I} = LocalOperator{I}(_project_terms(h, Bs), length(Bs))

# The per-term-shape machinery behind `adjoint(::OperatorSum)`. Every term must have total charge
# `unit(I)`: a charged term's adjoint carries the dual charge, which is a different object than a
# `Terms{I}` can hold. Only the spaces of the sites `h` actually touches are read.
function _adjoint_terms(h::Terms{I}, lat::AbstractLattice) where {I}
    out = Term{I}[]
    for t in h
        K = arity(t)
        if iszero(K)
            push!(out, Term{I}(Int[], ITOKey{I}[], conj(t.coeff)))
            continue
        end
        total(t) == unit(I) || throw(
            ArgumentError(
                "adjoint: the term on sites $(t.sites) has total charge $(total(t)); the adjoint " *
                    "of a charged term carries the dual charge $(dual(total(t))), so it is not a " *
                    "term of the same operator. Conjugate charge-neutral terms only."
            )
        )
        Vs = [lat[s] for s in t.sites]           # the sites' bases (or spaces)
        local_adj = _cached(_ADJOINT_CACHE, (t.keys, Vs)) do
            blk = _instantiate_term(Term{I}(collect(1:K), t.keys, ComplexF64(1)), Vs)
            return _project_local(removeunit(blk, 2K + 1)', Vs)
        end
        # placement drops the identity slots a basis projection can produce
        append!(out, scale(local_adj[t.sites...], conj(t.coeff)).terms)
    end
    return Terms{I}(out)
end

# `h'` is the natural thing to reach for, and a bare MethodError would not say what is missing.
Base.adjoint(::Terms) = throw(
    ArgumentError(
        "adjoint: the adjoint of an alphabet letter is a combination of the dual charge's letters " *
            "that only the physical space determines, so a bare Terms bag has no postfix `h'`. Put " *
            "it on a lattice first: `H = opsum(lat, h); H + H'`."
    )
)
