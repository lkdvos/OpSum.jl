# Phase 3: charge-augmented automaton (fusion-resolved terms)
# ===========================================================
# This file turns symbolic ITO terms (`GlobalOp` fields and `CoupledOp` couplings from Phase 2)
# into a *charge-augmented* `Trie` whose alphabet symbol is `(ITO, bond_charge_out, vertex)`
# instead of a bare `IrrepOperator`. The extra labels make the automaton block-diagonal in the
# bond charge, which is exactly what Phase 4 (per-sector bipartite matching + SVD) consumes.
#
# Design notes:
# * Term normal form is `FusedTerm(opstring, tree, coeff)`: a per-site vector of `IrrepOperator`
#   letters, a TensorKit caterpillar `FusionTree` over the local charges (its internal edges are
#   the virtual bond charges), and a scalar *reduced coefficient*.
# * Coupling channels are enumerated *canonically and boundedly* by delegating to TensorKit's
#   `fusiontrees` iterator — we never roll our own recoupling. The coupling invariants (fuses to
#   the requested total, unique channel for pairwise mult-free coupling) are `@assert`ed rather
#   than silently deduped (per the repo's "assert invariants, don't dedup" convention).
# * Idle sites are filled with a distinguished *pass-through identity* symbol: trivial charge
#   `unit(I)` with the sentinel index `n = 0`. It is NOT one of the enumerated `(c, n)` letters
#   (those start at `n = 1`) and NOT `one(A)` (which does not exist for `IrrepOperator`); it acts
#   as `id(V)` on instantiation.
# * The `Trie{K,V}` single-value-per-leaf contract is preserved by pushing the entire coupling
#   structure (bond charges + vertex labels) into the *key* `ITOKey`; the leaf *value* stays a
#   single scalar reduced coefficient.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, fusiontrees, sectortype, unit, dim
using .IrrepTensorOperators: IrrepOperator

# Pass-through identity symbol
# ----------------------------
"""
    passthrough(::Type{I}) where {I<:Sector}

The distinguished structural pass-through identity symbol for the ITO alphabet: the trivial
charge `unit(I)` tagged with the sentinel index `n = 0`. It fills idle sites of an operator
string, injecting trivial charge and acting as `id(V)`. It is deliberately *not* one of the
enumerated `(c, n)` alphabet letters (`n ≥ 1`) and not a type-level `one`.
"""
passthrough(::Type{I}) where {I <: Sector} = IrrepOperator{I}(unit(I), 0)

"""
    ispassthrough(op::IrrepOperator)

Whether `op` is the pass-through identity sentinel (index `n = 0`).
"""
ispassthrough(op::IrrepOperator) = op.n == 0

# Instantiation of the pass-through symbol as the structural identity `id(V)` is handled by the
# `n == 0` guard in the Phase-1 `instantiate(::IrrepOperator, V)` method (its charge leg is
# trivial, so it is the bare `V ← V` identity — consistent with the Phase-2 scalar-identity
# handling). Full instantiation of a fusion-resolved string into an MPO tensor is Phase 5.

# Augmented trie key
# ------------------
"""
    ITOKey{I<:Sector}

Alphabet symbol of the charge-augmented automaton. Bundles

* `op::IrrepOperator{I}` — the local ITO letter (or the pass-through sentinel on idle sites),
* `bond::I` — the bond charge *out* of this site (running fusion of the charges up to and
  including this site), and
* `vertex::Int` — the fusion-vertex multiplicity label at this site (`1` for multiplicity-free
  fusion).

Prefixes agreeing on the operator *and* the bond charge (*and* vertex) share a trie node;
differing couplings become distinct paths, so the trie is a direct sum of per-bond-sector FSMs.
"""
struct ITOKey{I <: Sector}
    op::IrrepOperator{I}
    bond::I
    vertex::Int
end

function Base.isless(x::ITOKey{I}, y::ITOKey{I}) where {I}
    x.op == y.op || return isless(x.op, y.op)
    x.bond == y.bond || return isless(x.bond, y.bond)
    return isless(x.vertex, y.vertex)
end
Base.:(==)(x::ITOKey{I}, y::ITOKey{I}) where {I} =
    x.op == y.op && x.bond == y.bond && x.vertex == y.vertex
Base.hash(x::ITOKey, h::UInt) = hash(x.vertex, hash(x.bond, hash(x.op, hash(:ITOKey, h))))

function Base.show(io::IO, k::ITOKey)
    return print(io, "ITOKey(", k.op, ", bond=", k.bond, ", μ=", k.vertex, ")")
end

# Term normal form
# ----------------
"""
    FusedTerm{I<:Sector, N, T}

A fusion-resolved operator term: `opstring` (one `IrrepOperator{I}` per site, pass-through on
idle sites), a caterpillar `tree::FusionTree{I,N}` over the local charges whose internal edges
are the virtual bond charges, and a scalar *reduced coefficient* `coeff::T`.
"""
struct FusedTerm{I <: Sector, N, T}
    opstring::Vector{IrrepOperator{I}}
    tree::FusionTree{I, N}
    coeff::T
end

"""
    bondcharges(f::FusionTree{I,N})

Running bond charges `b₁ … b_N` of the left-nested caterpillar tree: `b₁ = uncoupled[1]`,
`bₖ = innerlines[k-1]` for `2 ≤ k ≤ N-1`, `b_N = coupled`. Element `k` is the bond charge *out*
of site `k`.
"""
function bondcharges(f::FusionTree{I, N}) where {I, N}
    N == 0 && return I[]
    N == 1 && return I[f.coupled]
    return I[f.uncoupled[1], f.innerlines..., f.coupled]
end

_vlabel(v) = v isa Integer ? Int(v) : 1

"""
    vertexlabels(f::FusionTree{I,N})

Fusion-vertex multiplicity labels, one per site. Site 1 fuses trivially from the vacuum (label
`1`); site `k ≥ 2` carries `f.vertices[k-1]`.
"""
function vertexlabels(f::FusionTree{I, N}) where {I, N}
    N == 0 && return Int[]
    return Int[k == 1 ? 1 : _vlabel(f.vertices[k - 1]) for k in 1:N]
end

"""
    trie_key(ft::FusedTerm{I,N,T}) -> Vector{ITOKey{I}}

The charge-augmented key path (one `ITOKey` per site) spelled out by a fusion-resolved term.
"""
function trie_key(ft::FusedTerm{I, N, T}) where {I, N, T}
    bonds = bondcharges(ft.tree)
    verts = vertexlabels(ft.tree)
    return ITOKey{I}[ITOKey{I}(ft.opstring[i], bonds[i], verts[i]) for i in 1:N]
end

# Bounded, canonical coupling-channel enumeration
# -----------------------------------------------
"""
    caterpillar_trees(charges::NTuple{N,I}, total::I) -> Vector{FusionTree{I,N}}

All left-nested caterpillar fusion trees of the per-site `charges` fusing to `total`, in
TensorKit's canonical iteration order. This is the (finite, bounded) enumeration of the valid
intermediate bond-charge channels of a term — delegated to TensorKit rather than hand-rolled.
"""
function caterpillar_trees(charges::NTuple{N, I}, total::I) where {N, I <: Sector}
    N == 0 && throw(ArgumentError("empty operator string"))
    isd = ntuple(_ -> false, N)
    return collect(fusiontrees(charges, total, isd))
end

# Fusion resolution
# -----------------
# Extract the on-site (letter, scalar) terms of a `LocalOp`: a bare ITO letter, a scalar
# (mapped to `nothing` = pass-through), or a `Sum` distributing over both.
function _local_terms(op::LocalOp{T, A}) where {T, A <: IrrepOperator}
    o = variant(op)
    if o isa T
        return Tuple{Union{Nothing, A}, T}[(nothing, o)]
    elseif o isa A
        return Tuple{Union{Nothing, A}, T}[(o, one(T))]
    elseif o isa Sum
        out = Tuple{Union{Nothing, A}, T}[]
        for (k, v) in pairs(o.terms)
            for (letter, s) in _local_terms(k)
                push!(out, (letter, v * s))
            end
        end
        return out
    else
        throw(ArgumentError("cannot fusion-resolve on-site variant $(typeof(o)) (products/powers deferred)"))
    end
end

# Assemble one `FusedTerm` for a fixed opstring + total charge, asserting the coupling invariant.
function _resolve_string(opstring::Vector{IrrepOperator{I}}, total::I, coeff::T) where {I, T}
    N = length(opstring)
    charges = ntuple(i -> opstring[i].c, N)
    trees = caterpillar_trees(charges, total)
    @assert !isempty(trees) "charges $(charges) do not fuse to total charge $total"
    # Pairwise / single-site coupling in a multiplicity-free setting yields a unique channel.
    # Assert that invariant instead of silently picking or deduping one.
    @assert length(trees) == 1 "expected a unique coupling channel for charges $(charges) → $total, got $(length(trees)); multi-channel (GenericFusion / multi-body) coupling is deferred"
    return FusedTerm{I, N, T}(opstring, only(trees), coeff)
end

"""
    fusion_resolve(term, sites) -> Vector{FusedTerm}

Fusion-resolve a symbolic ITO `term` over the physical `sites` into normal form. Handles

* a `CoupledOp` (pairwise irrep coupling `couple(a, b; to)` / `a · b`), and
* a `GlobalOp` single-site field (`op[site]`), including scalar identities and `Sum`s that
  distribute into several fused terms.

The reduced coefficient folds together the per-operand scalar prefactors (e.g. spin reduced
matrix elements) and the coupling coefficient: `-√dim(c)` for a singlet target (matching the
Phase-2 Cartesian normalization) and `1` (unit Clebsch–Gordan) otherwise.
"""
function fusion_resolve(cop::CoupledOp, sites::AbstractVector{<:ElementarySpace})
    sa = variant(cop.a)
    sb = variant(cop.b)
    (sa isa SiteOp && length(sa.sites) == 1) ||
        throw(ArgumentError("fusion_resolve: first operand must be a single-site operator"))
    (sb isa SiteOp && length(sb.sites) == 1) ||
        throw(ArgumentError("fusion_resolve: second operand must be a single-site operator"))
    pa = only(sa.sites)
    pb = only(sb.sites)
    pa == pb && throw(ArgumentError("couple: the two operators must act on distinct sites"))

    at = _local_terms(sa.op)
    bt = _local_terms(sb.op)
    (length(at) == 1 && length(bt) == 1) ||
        throw(ArgumentError("fusion_resolve: coupled operands must be single ITO letters"))
    (la, sca) = only(at)
    (lb, scb) = only(bt)
    (la !== nothing && lb !== nothing) ||
        throw(ArgumentError("couple: both operands must carry an operator charge leg (not scalars)"))

    I = typeof(la.c)
    N = length(sites)
    total = cop.to::I

    # coupling coefficient: singlet → −√dim(cₐ) (Cartesian normalization), else unit CG.
    coupcoeff = total == unit(I) ? -sqrt(dim(la.c)) : one(ComplexF64)
    coeff = ComplexF64(sca) * ComplexF64(scb) * coupcoeff

    opstring = fill(passthrough(I), N)
    opstring[pa] = la
    opstring[pb] = lb
    return FusedTerm[_resolve_string(opstring, total, coeff)]
end

function fusion_resolve(g::GlobalOp{T, A, S}, sites::AbstractVector{<:ElementarySpace}) where {T, A <: IrrepOperator, S}
    o = variant(g)
    I = sectortype(A)
    N = length(sites)
    if o isa Sum
        out = FusedTerm[]
        for (k, v) in pairs(o.terms)
            for ft in fusion_resolve(k, sites)
                push!(out, FusedTerm(ft.opstring, ft.tree, v * ft.coeff))
            end
        end
        return out
    elseif o isa SiteOp
        @assert issorted(o.sites) && allunique(o.sites) "Sites must be sorted and unique."
        length(o.sites) <= 1 ||
            throw(ArgumentError("fusion_resolve: multi-site field embedding is not supported; express couplings with `couple`"))
        out = FusedTerm[]
        for (letter, coeff) in _local_terms(o.op)
            opstring = fill(passthrough(I), N)
            total = unit(I)
            if letter !== nothing
                isempty(o.sites) &&
                    throw(ArgumentError("fusion_resolve: charged on-site operator needs a site index"))
                opstring[only(o.sites)] = letter
                total = letter.c
            end
            push!(out, _resolve_string(opstring, total, ComplexF64(coeff)))
        end
        return out
    else
        throw(ArgumentError("fusion_resolve: unsupported GlobalOp variant $(typeof(o)) (couplings go through `couple`)"))
    end
end

# Charge-augmented trie
# ---------------------
"""
    irrep_trie(terms, sites) -> Trie{ITOKey{I}, ComplexF64}

Build the charge-augmented automaton trie for a collection of symbolic ITO `terms` (each a
`CoupledOp` or single-site `GlobalOp` field) over the physical `sites`. The trie key is the
`ITOKey` path of each fusion-resolved term; the leaf value is its reduced coefficient. Coincident
fused terms accumulate their coefficients (matching the dense `build_trie!` path); the per-term
coupling channels are asserted distinct during resolution.
"""
function irrep_trie(terms, sites::AbstractVector{<:ElementarySpace})
    I = sectortype(eltype(sites))
    K = ITOKey{I}
    T = ComplexF64
    root = Trie{K, T}()

    termlist = terms isa Union{CoupledOp, GlobalOp} ? (terms,) : terms
    for t in termlist
        resolved = fusion_resolve(t, sites)
        # single-term coupling invariant: the enumerated channels are distinct keys.
        keyset = Set{Vector{K}}()
        for ft in resolved
            key = trie_key(ft)::Vector{K}
            @assert !(key in keyset) "coupling invariant violated: duplicate fusion-resolved channel $key within a single term"
            push!(keyset, key)
            node = root
            for k in key
                node = haskey(node.children, k) ? node.children[k] : _add_child!(node, k)
            end
            node.value = isnothing(node.value) ? ft.coeff : node.value + ft.coeff
        end
    end
    return sortkeys!(root)
end

# Round-trip: reconstruct the fused-term list from trie paths
# -----------------------------------------------------------
"""
    trie_terms(trie::Trie{ITOKey{I}, T}) -> Vector{FusedTerm}

Reconstruct the fusion-resolved term list from the paths of a charge-augmented `trie` (inverse of
`irrep_trie` up to ordering and coefficient accumulation). The caterpillar tree is rebuilt from
the path's charges + total (unique in the multiplicity-free setting).
"""
function trie_terms(trie::Trie{ITOKey{I}, T}) where {I, T}
    out = FusedTerm[]
    for (key, coeff) in pairs(trie)
        opstring = IrrepOperator{I}[k.op for k in key]
        charges = ntuple(i -> opstring[i].c, length(opstring))
        total = last(key).bond
        tree = only(caterpillar_trees(charges, total))
        push!(out, FusedTerm(opstring, tree, coeff))
    end
    return out
end
