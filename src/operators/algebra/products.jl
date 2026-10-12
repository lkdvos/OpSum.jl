# Products of operators: `a[i] * b[j]`, and the on-site products `a[i] * b[i]` / `hop[i, i]` resolved
# in the sites' bases when they enter an `OperatorSum`. Design note: `research/operator-bases.md`
# §3.5, §3.6.
#
# A product is first written in *written order*: the factors' letters coupled as one caterpillar in
# that order, times a coefficient `cₓ c_y (-1)^{p(X) p(Y)}` (`p` the fermion parity of each factor's
# total charge; the graded sign `couple` uses, so `c†ᵢ * cᵢ` is `nᵢ`). Then (`_written!`):
#
# * distinct sites in site order: that is a `Term`;
# * distinct sites out of order, abelian: sort with one R-symbol per swapped pair (as `couple` does
#   when it inserts a leg; for fermions the anticommutation sign);
# * distinct sites out of order, non-abelian: permute the tree into site order and re-expand in
#   site-order caterpillars (`_recouple`), possibly several terms;
# * a repeated site: a `PendingTerm`, unevaluated until the sites' bases are known.

using TensorKit: Sector, unit, Rsymbol, Nsymbol, FusionTree, Vect, id, permute, numin, numout, domain,
    fusiontrees, BraidingStyle, SymmetricBraiding, FusionStyle, GenericFusion

# The total of a written-order caterpillar.
_writtentotal(keys::Vector{ITOKey{I}}) where {I} = isempty(keys) ? unit(I) : last(keys).bond

# Push a term written in any order, as `Term`s or a `PendingTerm`.
function _written!(out::Terms{I}, sites::Vector{Int}, keys::Vector{ITOKey{I}}, coeff::ComplexF64) where {I}
    allunique(sites) || return push!(out.pending, PendingTerm{I}(sites, keys, coeff))
    issorted(sites) && return push!(out.terms, Term{I}(sites, keys, coeff))
    _canreorder(I) || return append!(out.terms, _recouple(sites, keys, coeff))
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
    return push!(out.terms, Term{I}(sites[perm], outkeys, coeff))
end

# Re-coupling under a non-abelian symmetry: a caterpillar is the charge-only tensor
# `C₁ ⊗ … ⊗ C_K ← C_tot` with one nonzero entry at its tree (`_treetensor`). Permuting its legs into
# site order, or fusing two caterpillars, rewrites the term; reading the result back in the caterpillar
# basis (`_expand`) gives the new terms. Coefficients depend on charges and positions only: memoised.

# Multiplicity-free fusion keeps one tree per running-charge sequence (what a `Term` stores); symmetric
# braiding makes a leg permutation independent of crossings.
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

function _treetensor(charges::Vector{I}, bonds::Vector{I}) where {I}
    f = _tree_from_bonds(charges, bonds)
    tot = f.coupled
    X = zeros(ComplexF64, foldl(⊗, (Vect[I](c => 1) for c in charges)) ← Vect[I](tot => 1))
    X[f, FusionTree{I}((tot,), tot, (false,), ())] .= 1
    return X
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
function _recouple(sites::Vector{Int}, keys::Vector{ITOKey{I}}, coeff::ComplexF64) where {I}
    _checkrecouple(I)
    K = length(sites)
    perm = sortperm(sites)
    charges = I[k.op.c for k in keys]
    bonds = I[k.bond for k in keys]
    expansion = _cached(_RECOUPLE_CACHE, (:permute, charges, bonds, perm)) do
        return _expand(permute(_treetensor(charges, bonds), (Tuple(perm), (K + 1,))))
    end
    sorted, ops = sites[perm], [keys[j].op for j in perm]
    return Term{I}[
        Term{I}(sorted, ITOKey{I}[ITOKey{I}(ops[n], b[n], 1) for n in 1:K], coeff * x)
            for (b, x) in expansion
    ]
end

# Two written-order caterpillars fused to `tot`: caterpillars over the concatenated letters with their
# coefficients. When the first is neutral (or the symmetry abelian) the second's running charges just
# continue from the first's total.
function _concat(ka::Vector{ITOKey{I}}, kb::Vector{ITOKey{I}}, tot::I) where {I}
    ta = _writtentotal(ka)
    if ta == unit(I) || _canreorder(I)
        shifted = ITOKey{I}[ITOKey{I}(k.op, only(ta ⊗ k.bond), 1) for k in kb]
        return [(vcat(ka, shifted), 1.0 + 0im)]
    end
    isempty(kb) && return [(ka, 1.0 + 0im)]
    _checkrecouple(I)
    ca, cb = I[k.op.c for k in ka], I[k.op.c for k in kb]
    ba, bb = I[k.bond for k in ka], I[k.bond for k in kb]
    expansion = _cached(_RECOUPLE_CACHE, (:concat, ca, ba, cb, bb, tot)) do
        tb = last(bb)
        Y = _treetensor([ta, tb], [ta, tot])
        return _expand((_treetensor(ca, ba) ⊗ _treetensor(cb, bb)) * Y)
    end
    ops = vcat([k.op for k in ka], [k.op for k in kb])
    return [
        (ITOKey{I}[ITOKey{I}(ops[n], b[n], 1) for n in eachindex(ops)], x) for (b, x) in expansion
    ]
end

"""
    a * b      # a, b :: Terms (placed operators)

The operator product of two placed operators, in the order written; it distributes over their terms.
On different sites it is [`couple`](@ref) with the channel forced by the factors' charges, in any
site order: under an abelian symmetry the braiding phase is inserted (for fermions, the
anticommutation sign), under a non-abelian one the legs are re-coupled into site order, which can
turn one term into several.

When the factors share a site, `a[i] * b[i]`, the product is kept unevaluated and resolved in the
site's [`Basis`](@ref) when the operator enters an [`OperatorSum`](@ref). Until then the bag has no
normal form, so `length`, iteration and `≈` throw; `opsum(lat, a * b)` is the way to look at it.

`*` throws when the channel is not forced: a product of two charged operators whose totals fuse to
several sectors (non-abelian).
"""
function Base.:*(a::Terms{I}, b::Terms{I}) where {I}
    out = Terms{I}()
    for (sa, ka, ca) in _factors(a), (sb, kb, cb) in _factors(b)
        _product!(out, sa, ka, ca, sb, kb, cb)
    end
    return out
end

_factors(ts::Terms) = Iterators.flatten(
    (
        ((t.sites, t.keys, t.coeff) for t in ts.terms),
        ((p.sites, p.keys, p.coeff) for p in ts.pending),
    )
)

function _forcedproduct(ta::I, tb::I) where {I}
    cs = collect(ta ⊗ tb)
    (length(cs) == 1 && isone(Nsymbol(ta, tb, only(cs)))) || throw(
        ArgumentError(
            "`*` needs a forced channel, but under the $(FusionStyle(I)) symmetry $I the totals " *
                "$ta and $tb fuse to $(join(cs, ", ")); use `couple(...; to = c)` to name one"
        )
    )
    return only(cs)
end

function _product!(
        out::Terms{I}, sa::Vector{Int}, ka::Vector{ITOKey{I}}, ca::ComplexF64,
        sb::Vector{Int}, kb::Vector{ITOKey{I}}, cb::ComplexF64
    ) where {I}
    ta, tb = _writtentotal(ka), _writtentotal(kb)
    coeff = ca * cb
    _isfermionodd(ta) && _isfermionodd(tb) && (coeff = -coeff)
    sites = vcat(sa, sb)
    for (keys, x) in _concat(ka, kb, _forcedproduct(ta, tb))
        _written!(out, sites, keys, coeff * x)
    end
    return out
end

# Resolution
# ----------

const _PRODUCT_CACHE = Dict{Any, Any}()

# The terms a pending product stands for on `lat`: its written-order block on the sites it touches,
# projected into their bases and placed back. Memoised per (positions, keys, bases).
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
# (repeats allowed): the letters' embedded tensors multiplied in written order, charge legs kept in
# written order, then fused by the caterpillar `keys` carry.
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
    X = _treetensor(I[k.op.c for k in keys], I[k.bond for k in keys])
    W = foldl(⊗, (_space(B) for B in Bs))
    return T * (id(W) ⊗ X)                                              # W ← W ⊗ C_tot
end
