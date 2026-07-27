# Term-list symbolic algebra over ITOs (redesign)
# ================================================
# The ITO global algebra is a *sum of sparse, sited terms* (`TermSum` over `TermKey`), not an
# expression tree. Each term is a set of active sites, one ITO letter per active site, coupled by
# a caterpillar fusion tree to a total charge, weighted by a reduced coefficient. `op[site]`,
# `+`, `scale`/`*`, and `couple`/`·` all produce or merge terms; the automaton trie (see
# `irreptrie.jl`) is a derived index over the term list.
#
# Design notes:
# * Supported term arities: K = length(active sites) ∈ {0, 1, 2, …} — identity, single-site field,
#   and left-nested (caterpillar) coupling of arbitrarily many sites. On-site products/powers are
#   deferred; genuine multiplicity (`GenericFusion`, vertices > 1) is deferred.
# * Each term stores its caterpillar `FusionTree` explicitly (`TermKey.tree`): for K ≥ 3 the
#   intermediate ("inner line") charges are NOT determined by `(charges, total)`, so the tree is the
#   single source of truth. `total` is an accessor (`= tree.coupled`).
# * The scalar identity is materialized structurally as `c * id(V)`, never via `one(::Type{A})`.
# * The dense/Pauli `GlobalOp`/`LocalOp` pipeline is untouched; dense migrates to this later.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, fusiontrees, unit, dim, id,
    Vect, domain, permute, sectors, FusionStyle, UniqueFusion
import TensorKit: sectortype
using LinearAlgebra: LinearAlgebra
using .IrrepTensorOperators: IrrepOperator

export spin, scalarop, couple, TermSum

# sector type of the ITO alphabet
sectortype(::Type{IrrepOperator{I}}) where {I} = I

# LocalOp instantiation over ITOs (single-site materialization)
# -------------------------------------------------------------
"""
    instantiate(O::LocalOp{T,<:IrrepOperator}, V::ElementarySpace)

Materialize a single-site symbolic operator over the ITO alphabet: scalar → `c * id(V)`, bare
ITO → its Phase-1 tensor, `Sum` → scaled sum. On-site `Prod`/`Pow` are deferred.
"""
function instantiate(O::LocalOp{T, A}, V::ElementarySpace) where {T, A <: IrrepOperator}
    o = variant(O)
    if o isa T
        return o * id(V)
    elseif o isa A
        return instantiate(o, V)
    elseif o isa Sum
        isempty(o.terms) && throw(ArgumentError("cannot instantiate an empty Sum without a space"))
        return sum(pairs(o.terms)) do (k, v)
            return v * instantiate(k, V)
        end
    elseif o isa Prod || o isa Pow
        throw(ArgumentError("on-site products/powers of ITOs (fusion recoupling) are deferred"))
    else
        throw(ArgumentError("unsupported LocalOp variant for ITO instantiation: $(typeof(o))"))
    end
end

# Named accessors
# ---------------
"""
    spin(V::ElementarySpace)

The SU(2) rank-1 vector operator on a single-sector SU(2) space `V`, normalized so `spin(V)[i] ·
spin(V)[j]` densifies to the Cartesian `Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ`. Concretely
`spin(V) = √(s(s+1)(2s+1)) · IrrepOperator(SU2Irrep(1), 1)` (the reduced matrix element).
"""
function spin(V::ElementarySpace)
    sectortype(V) === SU2Irrep ||
        throw(ArgumentError("spin(V) requires an SU(2)-graded space, got $(sectortype(V))"))
    length(collect(sectors(V))) == 1 ||
        throw(ArgumentError("spin(V) requires a single SU(2) irrep sector"))
    s = only(sectors(V)).j
    scale = sqrt(s * (s + 1) * (2s + 1))
    A = IrrepOperator{SU2Irrep}
    return scale * LocalOp{ComplexF64, A}(A(SU2Irrep(1), 1))
end

"""
    scalarop(c::Number, V::ElementarySpace)
    scalarop(c::Number, ::Type{I<:Sector})

A scalar (identity) local operator `c·𝟙` over the ITO alphabet; instantiates structurally to
`c * id(V)` (the identity is not an alphabet letter).
"""
scalarop(c::Number, ::Type{I}) where {I} = LocalOp{ComplexF64, IrrepOperator{I}}(ComplexF64(c))
scalarop(c::Number, V::ElementarySpace) = scalarop(c, sectortype(V))

# Extract on-site (letter, scalar) terms of a LocalOp: bare letter, scalar (→ `nothing` =
# pass-through), or a `Sum` distributing over both.
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
        throw(ArgumentError("cannot resolve on-site variant $(typeof(o)) (products/powers deferred)"))
    end
end

# Caterpillar-tree constructors
# ------------------------------
# The 4-arg `FusionTree{I}(uncoupled, coupled, isdual, innerlines)` form is used everywhere (the
# 3-arg form is an abelian-only shortcut that throws for `MultipleFusion`); vertices default to `1`.

# K = 0 identity (empty) tree.
_idtree(::Type{I}) where {I <: Sector} = FusionTree{I}((), unit(I), (), ())

# K = 1 single-leg tree (uncoupled == coupled == c).
_leaftree(c::I) where {I <: Sector} = FusionTree{I}((c,), c, (false,), ())

# Rebuild a caterpillar tree from its per-active-position running bond charges `bonds`
# (`[c₁, innerlines…, total]`, length K) and vertex labels `verts` (`[1, μ₂…μ_K]`, length K) — the
# inverse of `bondcharges`/`vertexlabels`. Used by the trie/MPO round-trips.
function _tree_from_bonds(charges::AbstractVector{I}, bonds::AbstractVector{I}, verts::AbstractVector{Int}) where {I <: Sector}
    K = length(charges)
    K == 0 && return _idtree(I)
    unc = ntuple(i -> charges[i], K)
    isd = ntuple(_ -> false, K)
    inner = ntuple(i -> bonds[i + 1], max(0, K - 2))   # innerlines = bonds[2 : K-1]
    vtx = ntuple(i -> verts[i + 1], max(0, K - 1))     # vertices   = verts[2 : K]
    return FusionTree{I}(unc, bonds[K], isd, inner, vtx)
end

# Term normal form
# ----------------
"""
    TermKey{I<:Sector, S}

A fusion-resolved term (hashable key): `sites` (sorted, unique active sites), `ops` (one ITO letter
per active site, aligned with `sites`), and `tree::FusionTree{I}` — the left-nested (caterpillar)
coupling tree over the operator charges, whose `coupled` sector is the term's total charge
(accessor [`total`](@ref)). Storing the tree (rather than just `total`) is required for `K ≥ 3`
non-abelian terms, where the intermediate inner-line charges are not fixed by `(charges, total)`.
`K = length(sites)`: 0 = identity, 1 = field, ≥ 2 = caterpillar coupling. Restricted to
multiplicity-free fusion (all tree vertices `== 1`); `GenericFusion` is deferred.
"""
struct TermKey{I <: Sector, S}
    sites::Vector{S}
    ops::Vector{IrrepOperator{I}}
    tree::FusionTree{I}
    function TermKey{I, S}(
            sites::Vector{S}, ops::Vector{IrrepOperator{I}}, tree::FusionTree{I}
        ) where {I, S}
        @assert length(sites) == length(ops) == length(tree.uncoupled) "sites/ops/tree arity mismatch"
        @assert all(i -> ops[i].c == tree.uncoupled[i], eachindex(ops)) "tree uncoupled charges must match op charges"
        @assert all(isone, tree.vertices) "GenericFusion (multiplicity > 1) coupling is deferred"
        return new{I, S}(sites, ops, tree)
    end
end

"""
    total(k::TermKey)

The total (coupled) charge of a term: `k.tree.coupled`.
"""
total(k::TermKey) = k.tree.coupled

function Base.:(==)(x::TermKey{I, S}, y::TermKey{I, S}) where {I, S}
    return x.tree == y.tree && x.sites == y.sites && x.ops == y.ops
end
function Base.hash(x::TermKey, h::UInt)
    h = hash(:TermKey, hash(x.tree, h))
    for (s, o) in zip(x.sites, x.ops)
        h = hash(o, hash(s, h))
    end
    return h
end
function Base.show(io::IO, k::TermKey)
    return print(io, "TermKey(sites=", k.sites, ", ops=", k.ops, ", total=", total(k), ")")
end

"""
    TermSum{I<:Sector, S, T}

The ITO global algebra: a sum of `TermKey`s with reduced coefficients (`Dictionary{TermKey, T}`).
Built by `op[site]`, combined by `+`/`scale`/`*` and `couple`/`·`.
"""
struct TermSum{I <: Sector, S, T}
    terms::Dictionary{TermKey{I, S}, T}
end
TermSum{I, S, T}() where {I, S, T} = TermSum{I, S, T}(Dictionary{TermKey{I, S}, T}())

# Arithmetic
# ----------
function Base.:+(a::TermSum{I, S, T1}, b::TermSum{I, S, T2}) where {I, S, T1, T2}
    T = promote_type(T1, T2)
    d = Dictionary{TermKey{I, S}, T}()
    for (k, v) in pairs(a.terms)
        setwith!(+, d, k, convert(T, v))
    end
    for (k, v) in pairs(b.terms)
        setwith!(+, d, k, convert(T, v))
    end
    filter!(!iszero, d)
    return TermSum{I, S, T}(d)
end

function VectorInterface.scale(a::TermSum{I, S, T}, α::Number) where {I, S, T}
    Tn = promote_type(T, typeof(α))
    d = Dictionary{TermKey{I, S}, Tn}()
    for (k, v) in pairs(a.terms)
        iszero(α * v) || insert!(d, k, Tn(α * v))
    end
    return TermSum{I, S, Tn}(d)
end
Base.:*(α::Number, a::TermSum) = scale(a, α)
Base.:*(a::TermSum, α::Number) = scale(a, α)
Base.:/(a::TermSum, α::Number) = scale(a, inv(α))
Base.:-(a::TermSum) = scale(a, -1)
Base.:-(a::TermSum, b::TermSum) = a + (-b)

function Base.one(::TermSum{I, S, T}) where {I, S, T}
    d = Dictionary{TermKey{I, S}, T}()
    insert!(d, TermKey{I, S}(S[], IrrepOperator{I}[], _idtree(I)), one(T))
    return TermSum{I, S, T}(d)
end

function Base.show(io::IO, ts::TermSum)
    print(io, "TermSum(")
    join(io, ("$v * $k" for (k, v) in pairs(ts.terms)), " + ")
    return print(io, ")")
end

# op[site]  →  a (possibly distributed) TermSum
# ---------------------------------------------
function Base.getindex(O::LocalOp{T, A}, ind::S, inds::S...) where {T, A <: IrrepOperator, S}
    isempty(inds) ||
        throw(ArgumentError("multi-site placement of an ITO LocalOp is not supported; use `couple`"))
    I = sectortype(A)
    d = Dictionary{TermKey{I, S}, ComplexF64}()
    for (letter, coeff) in _local_terms(O)
        key = letter === nothing ?
            TermKey{I, S}(S[], IrrepOperator{I}[], _idtree(I)) :
            TermKey{I, S}(S[ind], IrrepOperator{I}[letter], _leaftree(letter.c))
        setwith!(+, d, key, ComplexF64(coeff))
    end
    return TermSum{I, S, ComplexF64}(d)
end

# Coupling
# --------
# Second operand of `dot`: a single-site charged term; returns `(site, op, coeff)`.
function _single_site_term(b::TermSum{I, S, T}, ctx) where {I, S, T}
    length(b.terms) == 1 || throw(ArgumentError("$ctx must be a single-term operator"))
    k, v = only(pairs(b.terms))
    (length(k.sites) == 1 && length(k.ops) == 1) ||
        throw(ArgumentError("$ctx must be a single-site charged operator"))
    return only(k.sites), only(k.ops), v
end

# Core of `couple`: extend every term of `a` by every single-site term of `b`, fusing to the total
# `target(ka, opb)` returns for that pair. Pairs whose charges cannot reach it are dropped, so the
# result may be empty — callers decide whether that is an error.
function _couple_terms(a::TermSum{I, S}, b::TermSum{I, S}, target) where {I, S}
    d = Dictionary{TermKey{I, S}, ComplexF64}()
    for (ka, va) in pairs(a.terms)
        isempty(ka.sites) && throw(
            ArgumentError("couple: every term of the first operand must carry at least one charged operator")
        )
        for (kb, vb) in pairs(b.terms)
            (length(kb.sites) == 1 && length(kb.ops) == 1) || throw(
                ArgumentError("couple: every term of the second operand must be a single-site charged operator")
            )
            sb, opb = only(kb.sites), only(kb.ops)
            sb in ka.sites && throw(ArgumentError("couple: operators must act on distinct sites"))
            maximum(ka.sites) < sb || throw(
                ArgumentError(
                    "couple: the second operand must act to the right of the first " *
                        "(left-nested caterpillar; out-of-order coupling is deferred)"
                )
            )

            # extend the caterpillar by one leg:  (total(a), opb.c) → tot, at a new vertex
            tot = target(ka, opb)
            f2s = collect(fusiontrees((total(ka), opb.c), tot, (false, false)))
            isempty(f2s) && continue    # this pair of charges cannot fuse to `tot`: drop it
            @assert length(f2s) == 1 "expected a unique coupling channel; multi-channel (GenericFusion) coupling is deferred"
            tree = TensorKit.join(ka.tree, only(f2s))

            key = TermKey{I, S}(S[ka.sites..., sb], IrrepOperator{I}[ka.ops..., opb], tree)
            setwith!(+, d, key, ComplexF64(va) * ComplexF64(vb))
        end
    end
    filter!(!iszero, d)
    return TermSum{I, S, ComplexF64}(d)
end

"""
    couple(a::TermSum, b::TermSum; to = unit(I))
    couple(a::TermSum, b::TermSum, cs::TermSum...; to = unit(I))   # abelian only

Left-nested (caterpillar) irrep coupling: extend the composite `a` (any K ≥ 1 sites, with its stored
coupling tree) by one single-site operator `b`, fusing the running total `total(a)` with `b`'s charge
to `to` at a new vertex. `to` defaults to the unit sector, which is what a term of a Hamiltonian
needs — pass it explicitly to build a charged object.

Each operand after the first must act to the **right** of every site before it (out-of-order
coupling needs F-moves and is deferred).

Both operands may be composite (several terms, e.g. from [`project`](@ref)): the coupling
distributes over every pair of terms, and pairs whose charges cannot fuse to `to` are dropped. It is
an error if *no* pair fuses. Each term of `a` must carry at least one charged operator, and each
term of `b` must be a single charged site.

For ``K ≥ 3`` the intermediate channels matter. Under an **abelian** symmetry (`UniqueFusion`: `U₁`,
`ℤₙ`, `FermionNumber`, `Trivial`, products thereof) every intermediate is forced by the charges, so
the variadic form does the whole chain for you:

```julia
couple(cd[1], c[2], cd[3], c[4])        # a charge-neutral four-fermion term
```

Under a non-abelian symmetry the intermediates are genuine freedom and the variadic form throws:
nest instead, naming each channel, so the choice is explicit and readable back off the term:

```julia
couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
```

This is the bare fusion coupler — it carries **no** normalization factor (reduced coeff `= va·vb`).
The Cartesian scalar-product convention lives in [`dot`](@ref), not here. Multi-channel
(`GenericFusion`) coupling and tree-structured (`via`) coupling are deferred.
"""
function couple(a::TermSum{I, S}, b::TermSum{I, S}; to = unit(I), via = nothing) where {I, S}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    isempty(a.terms) && throw(ArgumentError("couple: first operand has no terms"))
    isempty(b.terms) && throw(ArgumentError("couple: second operand has no terms"))
    tot = to::I

    out = _couple_terms(a, b, (_, _) -> tot)
    isempty(out.terms) &&
        throw(ArgumentError("couple: no pair of terms of the operands fuses to $tot"))
    return out
end

# Variadic coupling, abelian only: with `UniqueFusion` every intermediate charge is the single
# outcome of fusing the running total with the next operand's charge, so the whole caterpillar is
# fixed by the charges and there is nothing for the caller to choose. Fold left, letting each pair
# find its own intermediate, and constrain only the final total to `to`.
function couple(
        a::TermSum{I, S}, b::TermSum{I, S}, c::TermSum{I, S}, rest::TermSum{I, S}...;
        to = unit(I), via = nothing
    ) where {I, S}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    FusionStyle(I) isa UniqueFusion || throw(
        ArgumentError(
            "couple: variadic coupling needs an abelian symmetry (unique fusion), but $I has " *
                "$(FusionStyle(I)) — the intermediate channels are a real choice there. Nest " *
                "`couple` to name each one, e.g. couple(couple(a, b; to = b₂), c; to = t)"
        )
    )
    isempty(a.terms) && throw(ArgumentError("couple: first operand has no terms"))
    tot = to::I

    others = (b, c, rest...)
    acc = a
    for (i, nxt) in enumerate(others)
        isempty(nxt.terms) &&
            throw(ArgumentError("couple: operand $(i + 1) has no terms"))
        # intermediates are forced; only the last step has to land on `to`
        acc = if i == length(others)
            _couple_terms(acc, nxt, (_, _) -> tot)
        else
            _couple_terms(acc, nxt, (ka, opb) -> only(total(ka) ⊗ opb.c))
        end
        isempty(acc.terms) && throw(
            ArgumentError(
                i == length(others) ?
                    "couple: no chain of the operands fuses to $tot" :
                    "couple: no pair of terms of operands 1..$(i + 1) can be coupled"
            )
        )
    end
    return acc
end

"""
    dot(a::TermSum, b::TermSum)
    a · b

The Cartesian two-body scalar product of two single-site ITO operators: singlet coupling
`couple(a, b; to = unit(I))` times the Cartesian factor `-√dim(c)` (the identity
`Sᵢ·Sⱼ = -√3 [S⊗S]⁽⁰⁾` with `-√3 = -√dim(spin-1)`). Order-independent in the two sites. Distinct
from bare `couple(…; to = unit(I))`, which carries no such factor.

Unlike [`couple`](@ref) this does not distribute over composite operands: the Cartesian factor
`-√dim(c)` is per-letter, so it has no meaning for an operator mixing several charges. Use `couple`
for those.
"""
function LinearAlgebra.dot(a::TermSum{I}, b::TermSum{I}) where {I}
    sa, opa, _ = _single_site_term(a, "·: first operand")
    sb, _, _ = _single_site_term(b, "·: second operand")
    sa == sb && throw(ArgumentError("·: operators must act on distinct sites"))
    lo, hi = sa < sb ? (a, b) : (b, a)
    return scale(couple(lo, hi; to = unit(I)), -sqrt(dim(opa.c)))
end

# Dense-oracle materialization
# ----------------------------
"""
    instantiate(ts::TermSum, sites::AbstractVector{<:ElementarySpace})

Materialize the term-sum into a TensorKit `TensorMap` over the lattice `sites` (the dense oracle),
summing each term. Supports identity (K=0), single-site field (K=1), and left-nested (caterpillar)
coupling of any K ≥ 2 sites.
"""
function instantiate(ts::TermSum{I, S, T}, sites::AbstractVector{<:ElementarySpace}) where {I, S, T}
    isempty(ts.terms) && throw(ArgumentError("cannot instantiate an empty TermSum"))
    return sum(pairs(ts.terms)) do (k, v)
        return v * _instantiate_term(k, sites)
    end
end

function _instantiate_term(k::TermKey{I, S}, sites) where {I, S}
    N = length(sites)
    K = length(k.sites)
    if K == 0
        return foldl(⊗, (id(sites[j]) for j in 1:N))
    elseif K == 1
        return _embed_field(only(k.ops), only(k.sites), sites)
    else
        return _embed_caterpillar(k.ops, k.sites, k.tree, sites)
    end
end

# single charged field embedded on site `p`, identities elsewhere, charge leg to last domain slot
function _embed_field(op::IrrepOperator, p, sites)
    N = length(sites)
    loc = instantiate(op, sites[p])                # V_p ← V_p ⊗ V_c
    full = foldl(⊗, (j == p ? loc : id(sites[j]) for j in 1:N))
    cod = ntuple(identity, N)
    charge_global = N + (p + 1)
    dom_wo_charge = (ntuple(m -> N + m, p)..., ntuple(m -> N + p + 1 + m, N - p)...)
    dom = (dom_wo_charge..., charge_global)
    return permute(full, (cod, dom))
end

# Caterpillar-coupled K-site block (structural coupler; any reduced factor is in the coeff). The
# coupler `X` selects the *specific* fusion channel `tree` (setting its single reduced entry to 1 is
# a normalized isometry). Contraction of the K charge legs uses `permute` + morphism composition
# (bending the physical in-legs out and back), which generalizes to arbitrary K without a static
# `@tensor` expression.
function _embed_caterpillar(ops, positions, tree, sites)
    K = length(ops)
    I = typeof(tree.coupled)
    Os = [instantiate(ops[k], sites[positions[k]]) for k in 1:K]   # V ← V ⊗ Vc_k
    Vcs = [domain(Os[k])[2] for k in 1:K]
    tot = tree.coupled
    X = zeros(ComplexF64, foldl(⊗, Vcs) ← Vect[I](tot => 1))       # (Vc_1⊗…⊗Vc_K) ← Vect[tot]
    fcouple = FusionTree{I}((tot,), tot, (false,), ())
    X[tree, fcouple] .= 1

    P = foldl(⊗, Os)                                               # (o_1..o_K) ← (i_1,c_1,…,i_K,c_K)
    # bend physical in-legs into the codomain, leaving only the charge legs in the domain
    codP = (ntuple(k -> k, K)..., ntuple(k -> K + 2k - 1, K)...)   # o_1..o_K, i_1..i_K
    domP = ntuple(k -> K + 2k, K)                                  # c_1..c_K
    PX = permute(P, (codP, domP)) * X                              # (o.., i..) ← (tot)
    # split back into codomain (o_1..o_K) and domain (i_1..i_K, tot)
    Wblock = permute(PX, (ntuple(k -> k, K), (ntuple(k -> K + k, K)..., 2K + 1)))
    return _embed_block(Wblock, positions, sites)
end

# embed a K-site block on `positions` into the full lattice, reordering to site order with the
# total-charge leg last in the domain. `Wblock` codomain = (o over positions), domain = (i over
# positions, total-charge).
function _embed_block(Wblock, positions, sites)
    N = length(sites)
    K = length(positions)
    idle = [k for k in 1:N if !(k in positions)]
    full = isempty(idle) ? Wblock : Wblock ⊗ foldl(⊗, (id(sites[k]) for k in idle))

    cod_src = (positions..., idle...)
    p_cod = ntuple(j -> findfirst(==(j), cod_src), N)

    dom_src = (positions..., idle...)
    charge_global = N + (K + 1)                    # charge leg sits after the K active in-legs
    site_global(j) = let pos = findfirst(==(j), dom_src)
        pos <= K ? N + pos : N + pos + 1           # idle in-legs shift by 1 past the charge leg
    end
    p_dom = (ntuple(j -> site_global(j), N)..., charge_global)
    return permute(full, (p_cod, p_dom))
end
