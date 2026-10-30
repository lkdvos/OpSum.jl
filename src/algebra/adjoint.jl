# Hermitian conjugation of an `OperatorSum` (`H'`). The named operator sets live in stockbases.jl.

using TensorKit: ElementarySpace, Sector, sectortype, dual, removeunit

# No symbolic adjoint (a letter's adjoint carries the dual charge; the caterpillar coupler over dual
# charges is B-symbol data): materialise the term's `K`-site block, adjoint it, `project` it back.
# Memoised per distinct `(factors, bases)`: the key is the shape of a term together with the bases of its sites,
# so it is per term and lattice, not per basis, and stays here rather than in the `Basis`.

const _ADJOINT_CACHE = Dict{Any, Any}()

# Behind `adjoint(::OperatorSum)`; total charge must be `unit(I)`, since a charged adjoint carries the dual.
function adjoint_terms(h::Terms{I}, lat::AbstractLattice) where {I}
    out = Term{I}[]
    for t in h
        K = arity(t)
        if iszero(K)
            push!(out, Term{I}(Int[], Factor{I}[], conj(t.coeff)))
            continue
        end
        total(t) == unit(I) || throw(
            ArgumentError(
                "adjoint: the term on sites $(t.sites) has total charge $(total(t)); the adjoint " *
                    "of a charged term carries the dual charge $(dual(total(t))), so it is not a " *
                    "term of the same operator. Conjugate charge-neutral terms only."
            )
        )
        Vs = [lat[s] for s in t.sites]
        local_adj = cached(_ADJOINT_CACHE, (t.factors, Vs)) do
            blk = dense(Term{I}(collect(1:K), t.factors, ComplexF64(1)), Vs)
            return project_local(removeunit(blk, 2K + 1)', Vs)
        end
        # placement drops the identity slots a basis projection can produce
        append!(out, scale(local_adj[t.sites...], conj(t.coeff)).terms)
    end
    return Terms{I}(out)
end

Base.adjoint(::Terms) = throw(
    ArgumentError(
        "adjoint: the adjoint of an alphabet letter is a combination of the dual charge's letters " *
            "that only the physical space determines, so a bare Terms bag has no postfix `h'`. Put " *
            "it on a lattice first: `H = opsum(lat, h); H + H'`."
    )
)
