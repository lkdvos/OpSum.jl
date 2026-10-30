# Re-coupling of written-order caterpillars: a term whose legs were coupled in an order other than
# increasing site (`recouple`), and two caterpillars fused into one (`concat`). Used by placed `*` and
# `couple`, and by `written!` (products.jl).

using TensorKit
using TensorKit: Sector, unit, FusionTreeBlock, permute, BraidingStyle, SymmetricBraiding,
    FusionStyle, UniqueFusion, GenericFusion

# What a sector's fusion category supports. Caterpillars are multiplicity-free (`canfuse`: the bond
# charges determine the tree); a leg may be moved past others by a tree permutation (`canrecouple`) or,
# with unique fusion, by one scalar R-symbol per transposition (`canreorder`, the abelian case).
canfuse(::Type{I}) where {I <: Sector} = !(FusionStyle(I) isa GenericFusion)
canrecouple(::Type{I}) where {I <: Sector} = canfuse(I) && BraidingStyle(I) isa SymmetricBraiding
canreorder(::Type{I}) where {I <: Sector} =
    FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding

# Throw unless `can(I)`; `what` says what was attempted.
requirefusion(can, ::Type{I}, what::AbstractString) where {I} = can(I) || throw(
    ArgumentError("$what (got $I with $(FusionStyle(I)) fusion and $(BraidingStyle(I)) braiding)")
)

# Re-coupling under a non-abelian symmetry. A caterpillar is a left-nested fusion tree of the letters'
# charges; the running bond charges *are* its inner lines. Permuting the legs, or fusing two caterpillars,
# rewrites it as a combination of trees (TensorKit's `permute` and `merge` on fusion trees), whose inner
# lines give the new terms. Needs one tree per running-charge sequence (multiplicity-free) and
# crossing-independent permutations.

"""
Coefficients of a re-coupling below this are rounding noise of the F- and R-symbol products, not
terms: those trees are dropped (a tree in the expansion has coefficient of order one or is structurally absent).
"""
const RECOUPLE_ATOL = 1.0e-13

_checkrecouple(::Type{I}) where {I} = requirefusion(
    canrecouple, I, "re-coupling the legs is not supported: write the factors in increasing site order"
)

# The running charges of each tree of an expansion `tree => coefficient`, above the noise level.
function _expansion(::Type{I}, trees) where {I}
    out = Tuple{Vector{I}, ComplexF64}[]
    for (f, x) in trees
        abs(x) > RECOUPLE_ATOL && push!(out, (bondcharges(f), ComplexF64(x)))
    end
    return out
end

# The caterpillar `charges`, `bonds` with its legs permuted by `perm` (`perm[n]` is the old position of the
# new leg `n`), as trees. `permute` on a single tree needs unique fusion, so the tree sits in the block of
# all trees of the same legs coupled to its total (the domain leg), whose transformation `U` gives the row.
function _permuted(charges::Vector{I}, bonds::Vector{I}, perm::Vector{Int}) where {I}
    K = length(charges)
    tot = last(bonds)
    isdual = (ntuple(_ -> false, K), (false,))
    block = FusionTreeBlock{I}((Tuple(charges), (tot,)), isdual)
    dst, U = permute(block, (Tuple(perm), (K + 1,)))
    f = tree_from_bonds(charges, bonds)
    i = findfirst(t -> t[1] == f, block.trees)
    i === nothing && invariant_violation("caterpillar tree not found in its block")
    return [dst.trees[j][1] => U[j, i] for j in eachindex(dst.trees)]
end

const _RECOUPLE_CACHE = Dict{Any, Any}()

# A written-order caterpillar on distinct sites, as site-order terms.
function recouple(sites::Vector{Int}, keys::Vector{Factor{I}}, coeff::ComplexF64) where {I}
    _checkrecouple(I)
    K = length(sites)
    perm = sortperm(sites)
    charges, bonds = lettercharges(keys), bondcharges(keys)
    terms = cached(_RECOUPLE_CACHE, (:permute, charges, bonds, perm)) do
        return _expansion(I, _permuted(charges, bonds, perm))
    end
    sorted, ops = sites[perm], letters(keys)[perm]
    return Term{I}[
        Term{I}(sorted, Factor{I}[Factor{I}(ops[n], b[n]) for n in 1:K], coeff * x)
            for (b, x) in terms
    ]
end

# Two written-order caterpillars fused to `tot`, as `(keys, coeff)` over the concatenated letters. When
# the first is neutral (or the symmetry abelian) the second's running charges just continue.
function concat(ka::Vector{Factor{I}}, kb::Vector{Factor{I}}, tot::I) where {I}
    ta = total(ka)
    if ta == unit(I) || canreorder(I)
        shifted = Factor{I}[Factor{I}(k.letter, only(ta ⊗ k.bond)) for k in kb]
        return [(vcat(ka, shifted), 1.0 + 0im)]
    end
    isempty(kb) && return [(ka, 1.0 + 0im)]
    _checkrecouple(I)
    ca, cb, ba, bb = lettercharges(ka), lettercharges(kb), bondcharges(ka), bondcharges(kb)
    terms = cached(_RECOUPLE_CACHE, (:concat, ca, ba, cb, bb, tot)) do
        return _expansion(I, TensorKit.merge(tree_from_bonds(ca, ba), tree_from_bonds(cb, bb), tot))
    end
    ops = vcat(letters(ka), letters(kb))
    return [
        (Factor{I}[Factor{I}(ops[n], b[n]) for n in eachindex(ops)], x) for (b, x) in terms
    ]
end
