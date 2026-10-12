# Hermitian conjugation of an `OperatorSum` (`H'`). The named operator sets live in stockbases.jl.

using TensorKit: ElementarySpace, Sector, sectortype, dual, removeunit

# There is no cheap symbolic adjoint: a letter's adjoint carries the dual charge, and re-expressing
# the caterpillar coupler over dual charges is B-symbol data. So materialise the term's `K`-site
# block, take the adjoint there and `project` it back (faithful by construction), memoised per
# distinct `(keys, spaces)`.

const _ADJOINT_CACHE = Dict{Any, Any}()

# A `K`-site block in the letters of its sites, each slot in its site's basis (throws if outside the span).
_project_local(h, Bs::AbstractVector{<:Basis{I}}) where {I} = LocalOperator{I}(_project_terms(h, Bs), length(Bs))

# Behind `adjoint(::OperatorSum)`. Terms must have total charge `unit(I)`: a charged term's adjoint
# carries the dual charge, which a `Terms{I}` cannot hold.
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

Base.adjoint(::Terms) = throw(
    ArgumentError(
        "adjoint: the adjoint of an alphabet letter is a combination of the dual charge's letters " *
            "that only the physical space determines, so a bare Terms bag has no postfix `h'`. Put " *
            "it on a lattice first: `H = opsum(lat, h); H + H'`."
    )
)
