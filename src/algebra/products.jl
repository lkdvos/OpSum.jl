# Products of operators: `a[i] * b[j]`, and the on-site products `a[i] * b[i]` / `hop[i, i]` resolved in
# the sites' bases when they enter an `OperatorSum`.
#
# A product is first written in *written order*: the factors' letters coupled as one caterpillar in that
# order, times `cₓ c_y (-1)^{p(X) p(Y)}` (`p` the fermion parity of the total charge; the graded sign
# `couple` uses, so `c†ᵢ * cᵢ` is `nᵢ`). Then `written!` makes it a `Term` (distinct sites; out of order:
# one R-symbol per swapped pair when abelian, `recouple` otherwise) or a `PendingTerm` (a repeated site).

using TensorKit: Sector, unit, Rsymbol, Nsymbol, FusionTree, Vect, id, permute, numin, numout, domain,
    fusiontrees, BraidingStyle, SymmetricBraiding, FusionStyle, GenericFusion

# Push a term written in any order, as `Term`s or a `PendingTerm`.
function written!(out::TermBag{I}, sites::Vector{Int}, keys::Vector{Factor{I}}, coeff::ComplexF64) where {I}
    allunique(sites) || return push!(out.pending, PendingTerm{I}(sites, keys, coeff))
    issorted(sites) && return push!(out.terms, Term{I}(sites, keys, coeff))
    canreorder(I) || return append!(out.terms, recouple(sites, keys, coeff))
    # abelian: one R-symbol per pair that swaps, then the running charges in site order
    K = length(sites)
    for j in 1:K, k in (j + 1):K
        sites[j] > sites[k] || continue
        a, b = keys[j].letter.c, keys[k].letter.c
        coeff *= Rsymbol(a, b, only(a ⊗ b))
    end
    perm = sortperm(sites)
    outkeys = Vector{Factor{I}}(undef, K)
    bond = unit(I)
    for (n, j) in enumerate(perm)
        op = keys[j].letter
        bond = only(bond ⊗ op.c)
        outkeys[n] = Factor{I}(op, bond)
    end
    return push!(out.terms, Term{I}(sites[perm], outkeys, coeff))
end

"""
    a * b      # a, b :: Terms (placed operators)

The operator product of two placed operators, in the order written, distributing over their terms. On
different sites it is [`couple`](@ref) with the channel forced by the factors' charges, in any site
order (braiding phase for abelian symmetries, re-coupling into site order otherwise, which can give
several terms). A shared site, `a[i] * b[i]`, stays unevaluated until the operator enters an
[`OperatorSum`](@ref); until then the bag has no normal form, so `length`, iteration and `≈` throw.

`*` throws when the channel is not forced (two charged totals fusing to several sectors).
"""
function Base.:*(a::Terms{I}, b::Terms{I}) where {I}
    out = TermBag{I}()
    for (sa, ka, ca) in _factors(a), (sb, kb, cb) in _factors(b)
        ta, tb = total(ka), total(kb)
        cs = collect(ta ⊗ tb)
        (length(cs) == 1 && isone(Nsymbol(ta, tb, only(cs)))) || throw(
            ArgumentError(
                "`*` needs a forced channel, but under the $(FusionStyle(I)) symmetry $I the totals " *
                    "$ta and $tb fuse to $(join(cs, ", ")); use `couple(...; to = c)` to name one"
            )
        )
        coeff = ca * cb * gradedsign(ta, tb)
        for (keys, x) in concat(ka, kb, only(cs))
            written!(out, vcat(sa, sb), keys, coeff * x)
        end
    end
    return Terms{I}(out)
end

_factors(ts::Terms) = Iterators.flatten(
    (
        ((t.sites, t.factors, t.coeff) for t in ts.terms),
        ((p.sites, p.factors, p.coeff) for p in ts.pending),
    )
)

# The charge-only tensor `C₁ ⊗ … ⊗ C_K ← C_tot` with one nonzero entry, at the caterpillar tree of `bonds`.
function _treetensor(charges::Vector{I}, bonds::Vector{I}) where {I}
    f = tree_from_bonds(charges, bonds)
    tot = f.coupled
    X = zeros(ComplexF64, foldl(⊗, (Vect[I](c => 1) for c in charges)) ← Vect[I](tot => 1))
    X[f, FusionTree{I}((tot,), tot, (false,), ())] .= 1
    return X
end

# Resolution
# ----------

const _PRODUCT_CACHE = Dict{Any, Any}()

# The terms a pending product stands for on `lat`: its written-order block on the sites it touches,
# projected into their bases and placed back (memoised).
function resolve(p::PendingTerm{I}, lat::AbstractLattice) where {I}
    U = sort!(unique(p.sites))
    pos = Int[searchsortedfirst(U, s) for s in p.sites]
    Bs = [lat[u] for u in U]
    local_op = cached(_PRODUCT_CACHE, (pos, p.factors, Bs)) do
        return project_local(written_block(p.factors, pos, Bs), Bs)
    end
    return scale(local_op[U...], p.coeff).terms
end

resolve(ps::Vector{PendingTerm{I}}, lat::AbstractLattice) where {I} =
    reduce(vcat, (resolve(p, lat) for p in ps); init = Term{I}[])

# The written-order caterpillar of letters at positions `pos` of the block over `Bs` (repeats allowed):
# the embedded tensors multiplied in written order, charge legs kept in that order, then fused by `keys`.
function written_block(keys::Vector{Factor{I}}, pos::Vector{Int}, Bs) where {I}
    m = length(Bs)
    T = nothing
    for (key, p) in zip(keys, pos)
        E = embed_letter(key.letter, p, Bs)                                 # W ← W ⊗ C
        if T === nothing
            T = E
        else
            nk = numin(T) - m                                           # charge legs so far
            Cs = foldl(⊗, (domain(T)[m + j] for j in 1:nk))
            R = T * (E ⊗ id(Cs))                                        # W ← W ⊗ C ⊗ Cs
            dom = (ntuple(j -> m + j, m)..., ntuple(j -> 2m + 1 + j, nk)..., 2m + 1)
            T = permute(R, (ntuple(identity, m), dom))                  # W ← W ⊗ Cs ⊗ C
        end
    end
    X = _treetensor(lettercharges(keys), bondcharges(keys))
    W = foldl(⊗, (space(B) for B in Bs))
    return T * (id(W) ⊗ X)                                              # W ← W ⊗ C_tot
end
