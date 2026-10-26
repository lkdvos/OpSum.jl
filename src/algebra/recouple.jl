# Re-coupling of written-order caterpillars: a term whose legs were coupled in an order other than
# increasing site (`recouple`), and two caterpillars fused into one (`concat`). Used by placed `*` and
# `couple`, and by `written!` (products.jl).

using TensorKit: Sector, unit, FusionTree, Vect, permute, fusiontrees, BraidingStyle, SymmetricBraiding,
    FusionStyle, GenericFusion

# The total of a written-order caterpillar.
writtentotal(keys::Vector{Factor{I}}) where {I} = isempty(keys) ? unit(I) : last(keys).bond

# Re-coupling under a non-abelian symmetry: a caterpillar is the charge-only tensor `C₁ ⊗ … ⊗ C_K ← C_tot`
# with one nonzero entry at its tree (`treetensor`). Permuting its legs or fusing two caterpillars
# rewrites the term; reading it back in the caterpillar basis (`_expand`) gives the new terms (memoised).
# Needs one tree per running-charge sequence (multiplicity-free) and crossing-independent permutations.
_canrecouple(::Type{I}) where {I <: Sector} =
    !(FusionStyle(I) isa GenericFusion) && BraidingStyle(I) isa SymmetricBraiding

function _checkrecouple(::Type{I}) where {I}
    return _canrecouple(I) || throw(
        ArgumentError(
            "re-coupling the legs of a $(FusionStyle(I)) coupling with $(BraidingStyle(I)) braiding " *
                "is not supported: write the factors in increasing site order"
        )
    )
end

# A charge-only tensor in the caterpillar basis: the running charges of each tree and its coefficient.
function _expand(X)
    out = Tuple{Vector{sectortype(X)}, ComplexF64}[]
    for (f₁, f₂) in fusiontrees(X)
        x = only(X[f₁, f₂])
        abs(x) > 1.0e-13 && push!(out, (bondcharges(f₁), x))
    end
    return out
end

const _RECOUPLE_CACHE = Dict{Any, Any}()

# A written-order caterpillar on distinct sites, as site-order terms.
function recouple(sites::Vector{Int}, keys::Vector{Factor{I}}, coeff::ComplexF64) where {I}
    _checkrecouple(I)
    K = length(sites)
    perm = sortperm(sites)
    charges = I[k.letter.c for k in keys]
    bonds = I[k.bond for k in keys]
    expansion = cached(_RECOUPLE_CACHE, (:permute, charges, bonds, perm)) do
        return _expand(permute(treetensor(charges, bonds), (Tuple(perm), (K + 1,))))
    end
    sorted, ops = sites[perm], [keys[j].letter for j in perm]
    return Term{I}[
        Term{I}(sorted, Factor{I}[Factor{I}(ops[n], b[n]) for n in 1:K], coeff * x)
            for (b, x) in expansion
    ]
end

# Two written-order caterpillars fused to `tot`, as `(keys, coeff)` over the concatenated letters. When
# the first is neutral (or the symmetry abelian) the second's running charges just continue.
function concat(ka::Vector{Factor{I}}, kb::Vector{Factor{I}}, tot::I) where {I}
    ta = writtentotal(ka)
    if ta == unit(I) || canreorder(I)
        shifted = Factor{I}[Factor{I}(k.letter, only(ta ⊗ k.bond)) for k in kb]
        return [(vcat(ka, shifted), 1.0 + 0im)]
    end
    isempty(kb) && return [(ka, 1.0 + 0im)]
    _checkrecouple(I)
    ca, cb = I[k.letter.c for k in ka], I[k.letter.c for k in kb]
    ba, bb = I[k.bond for k in ka], I[k.bond for k in kb]
    expansion = cached(_RECOUPLE_CACHE, (:concat, ca, ba, cb, bb, tot)) do
        tb = last(bb)
        Y = treetensor([ta, tb], [ta, tot])
        return _expand((treetensor(ca, ba) ⊗ treetensor(cb, bb)) * Y)
    end
    ops = vcat([k.letter for k in ka], [k.letter for k in kb])
    return [
        (Factor{I}[Factor{I}(ops[n], b[n]) for n in eachindex(ops)], x) for (b, x) in expansion
    ]
end
