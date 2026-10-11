# Products of operators: `a[i] * b[j]`, and the on-site products `a[i] * b[i]` / `hop[i, i]` that are
# resolved in the sites' bases when they enter an `OperatorSum`. Design note:
# `research/operator-bases.md` §3.1 (item 7) and §3.5.
#
# Every product is first written down in *written order*: the factors' letters one after the other,
# coupled as one caterpillar in that order, times a coefficient. Its operator is `coeff · raw`, with
# `raw` the written-order caterpillar — for letters on increasing, distinct sites exactly what
# `_instantiate_term` materialises, and in general `_written_block`. From there:
#
# * distinct sites, already in site order: that *is* a `Term`;
# * distinct sites out of order, abelian symmetry: the letters are sorted with one R-symbol per pair
#   that swaps, the same move `couple` makes when it inserts a leg (for fermions, the anticommutation
#   sign);
# * distinct sites out of order, non-abelian: re-coupling needs F-moves, which is deferred;
# * a site that repeats: a `PendingTerm`, kept unevaluated until the sites' bases are known.
#
# The coefficient of `X * Y` is `cₓ c_y (-1)^{p(X) p(Y)}`, `p` the fermion parity of each factor's
# total charge: the graded product, the same sign `couple` applies, which is what makes the written
# order the physical operator order (`c†ᵢ * cᵢ` is `nᵢ`).

using TensorKit: Sector, unit, Rsymbol, FusionTree, Vect, id, permute, numin, numout, domain

# The total of a written-order caterpillar.
_writtentotal(keys::Vector{ITOKey{I}}) where {I} = isempty(keys) ? unit(I) : last(keys).bond

# A term written in an arbitrary order, as a `Term` (sorted) or a `PendingTerm` (a site repeats).
function _written(sites::Vector{Int}, keys::Vector{ITOKey{I}}, coeff::ComplexF64) where {I}
    allunique(sites) || return PendingTerm{I}(sites, keys, coeff)
    issorted(sites) && return Term{I}(sites, keys, coeff)
    _canreorder(I) || throw(
        ArgumentError(
            "the operator acts on sites $sites in that order, and reordering the legs of a " *
                "$(FusionStyle(I)) coupling needs F-moves, which is deferred: write the factors " *
                "in increasing site order, or let them share a site"
        )
    )
    # abelian: one R-symbol per pair that swaps, then the running charges in site order
    K = length(sites)
    for j in 1:K, k in (j + 1):K
        sites[j] > sites[k] || continue
        a, b = keys[j].op.c, keys[k].op.c
        coeff *= Rsymbol(a, b, only(a ⊗ b))
    end
    perm = sortperm(sites)
    outkeys = Vector{ITOKey{I}}(undef, K)
    bond = unit(I)
    for (n, j) in enumerate(perm)
        op = keys[j].op
        bond = only(bond ⊗ op.c)
        outkeys[n] = ITOKey{I}(op, bond, 1)
    end
    return Term{I}(sites[perm], outkeys, coeff)
end

_push!(out::Terms{I}, t::Term{I}) where {I} = push!(out.terms, t)
_push!(out::Terms{I}, p::PendingTerm{I}) where {I} = push!(out.pending, p)

"""
    a * b      # a, b :: Terms (placed operators)

The operator product of two placed operators, in the order written; it distributes over their terms.
On different sites it is [`couple`](@ref) with the channel forced by the factors' charges — the
product of their totals under an abelian symmetry, the unit sector when both factors are neutral —
in any site order under an abelian symmetry, and in site order (or with the factors' supports not
interleaved) under a non-abelian one.

When the factors share a site, `a[i] * b[i]`, the product is kept unevaluated: it is resolved in the
site's [`Basis`](@ref) when the operator enters an [`OperatorSum`](@ref), by multiplying the factors'
tensors and projecting back (memoised per term shape). Until then the bag has no normal form, so
`length`, iteration and `≈` throw; `opsum(lat, a * b)` is the way to look at it.

`*` throws when the channel is not forced: a product of two charged operators under a non-abelian
symmetry, whose total could be any of several sectors. Model code built from neutral operators never
meets that.
"""
function Base.:*(a::Terms{I}, b::Terms{I}) where {I}
    out = Terms{I}()
    for (sa, ka, ca) in _factors(a), (sb, kb, cb) in _factors(b)
        _push!(out, _product(sa, ka, ca, sb, kb, cb))
    end
    return out
end

_factors(ts::Terms) = Iterators.flatten(
    (
        ((t.sites, t.keys, t.coeff) for t in ts.terms),
        ((p.sites, p.keys, p.coeff) for p in ts.pending),
    )
)

function _product(
        sa::Vector{Int}, ka::Vector{ITOKey{I}}, ca::ComplexF64,
        sb::Vector{Int}, kb::Vector{ITOKey{I}}, cb::ComplexF64
    ) where {I}
    ta, tb = _writtentotal(ka), _writtentotal(kb)
    if _canreorder(I)
        # abelian: every channel is forced; `b`'s running charges continue from `a`'s total
        keys = vcat(ka, ITOKey{I}[ITOKey{I}(k.op, only(ta ⊗ k.bond), 1) for k in kb])
        sites = vcat(sa, sb)
    else
        (ta == unit(I) && tb == unit(I)) || throw(
            ArgumentError(
                "`*` needs a forced channel: under the $(FusionStyle(I)) symmetry $I that means " *
                    "neutral factors, got totals $ta and $tb; use `couple(...; to = c)` to name one"
            )
        )
        # neutral factors on disjoint supports commute, so one lying wholly to the left goes first
        if !isempty(sa) && !isempty(sb) && maximum(sb) < minimum(sa)
            sites, keys = vcat(sb, sa), vcat(kb, ka)
        else
            sites, keys = vcat(sa, sb), vcat(ka, kb)
        end
    end
    coeff = ca * cb
    _isfermionodd(ta) && _isfermionodd(tb) && (coeff = -coeff)
    return _written(sites, keys, coeff)
end

# Resolution
# ----------

const _PRODUCT_CACHE = Dict{Any, Any}()

# The terms a pending product stands for on the lattice `lat`: its written-order block on the sites
# it touches, projected into their bases (or matrix units, on a lattice of spaces) and placed back.
# Memoised per shape — positions within the block, letters with their bonds, and the bases.
function _resolve(p::PendingTerm{I}, lat::AbstractLattice) where {I}
    U = sort!(unique(p.sites))
    pos = Int[searchsortedfirst(U, s) for s in p.sites]
    Bs = [lat[u] for u in U]
    local_op = _cached(_PRODUCT_CACHE, (pos, p.keys, Bs)) do
        return _project_local(_written_block(p.keys, pos, Bs), Bs)
    end
    return scale(local_op[U...], p.coeff).terms
end

_resolve(ps::Vector{PendingTerm{I}}, lat::AbstractLattice) where {I} =
    reduce(vcat, (_resolve(p, lat) for p in ps); init = Term{I}[])

# The written-order caterpillar of single-site letters at positions `pos` of the block over `Bs`
# (repeats allowed): the product of the letters' embedded tensors in written order, their charge legs
# kept in written order, then fused by the caterpillar whose running charges `keys` carry. For
# distinct increasing positions this is `_instantiate_term`'s tensor.
function _written_block(keys::Vector{ITOKey{I}}, pos::Vector{Int}, Bs) where {I}
    m = length(Bs)
    T = nothing
    for (key, p) in zip(keys, pos)
        E = _embed_field(key.op, p, Bs)                                 # W ← W ⊗ C
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
    n = length(keys)
    charges = [k.op.c for k in keys]
    tree = _tree_from_bonds(charges, [k.bond for k in keys], ones(Int, n))
    tot = tree.coupled
    X = zeros(ComplexF64, foldl(⊗, (Vect[I](c => 1) for c in charges)) ← Vect[I](tot => 1))
    X[tree, FusionTree{I}((tot,), tot, (false,), ())] .= 1
    W = foldl(⊗, (_space(B) for B in Bs))
    return T * (id(W) ⊗ X)                                              # W ← W ⊗ C_tot
end
