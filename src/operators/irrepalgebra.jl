# Term-list symbolic algebra over ITOs (redesign)
# ================================================
# The ITO global algebra is a *sum of sparse, sited terms* (`TermSum` over `TermKey`), not an
# expression tree. Each term is a set of active sites, one ITO letter per active site, coupled by
# a caterpillar fusion tree to a total charge, weighted by a reduced coefficient. `op[site]`,
# `+`, `scale`/`*`, and `couple`/`·` all produce or merge terms; the automaton trie (see
# `irreptrie.jl`) is a derived index over the term list.
#
# Design notes:
# * Supported term arities: K = length(active sites) ∈ {0, 1, 2} — identity, single-site field,
#   pairwise coupling. Multi-body (>2) coupling and on-site products/powers are deferred.
# * The coupling channel is stored implicitly as the `total` charge; the caterpillar tree is
#   re-derived from `(charges, total)` (unique in the multiplicity-free pairwise setting).
# * The scalar identity is materialized structurally as `c * id(V)`, never via `one(::Type{A})`.
# * The dense/Pauli `GlobalOp`/`LocalOp` pipeline is untouched; dense migrates to this later.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, fusiontrees, sectortype, unit, dim, id,
    Vect, @tensor, domain, numin, permute, sectors
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

# Term normal form
# ----------------
"""
    TermKey{I<:Sector, S}

A fusion-resolved term (hashable key): `sites` (sorted, unique active sites), `ops` (one ITO
letter per active site, aligned with `sites`), and `total::I` (the charge the active charges fuse
to). The caterpillar coupling tree is derived from `(charges, total)` on demand.
`K = length(sites) ∈ {0,1,2}`: identity, field, or pairwise coupling.
"""
struct TermKey{I <: Sector, S}
    sites::Vector{S}
    ops::Vector{IrrepOperator{I}}
    total::I
end

function Base.:(==)(x::TermKey{I, S}, y::TermKey{I, S}) where {I, S}
    return x.total == y.total && x.sites == y.sites && x.ops == y.ops
end
function Base.hash(x::TermKey, h::UInt)
    h = hash(:TermKey, hash(x.total, h))
    for (s, o) in zip(x.sites, x.ops)
        h = hash(o, hash(s, h))
    end
    return h
end
function Base.show(io::IO, k::TermKey)
    return print(io, "TermKey(sites=", k.sites, ", ops=", k.ops, ", total=", k.total, ")")
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
    insert!(d, TermKey{I, S}(S[], IrrepOperator{I}[], unit(I)), one(T))
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
            TermKey{I, S}(S[], IrrepOperator{I}[], unit(I)) :
            TermKey{I, S}(S[ind], IrrepOperator{I}[letter], letter.c)
        setwith!(+, d, key, ComplexF64(coeff))
    end
    return TermSum{I, S, ComplexF64}(d)
end

# Coupling
# --------
function _single_charged_term(a::TermSum{I, S, T}, ctx) where {I, S, T}
    length(a.terms) == 1 || throw(ArgumentError("$ctx must be a single-term operator"))
    k, v = only(pairs(a.terms))
    (length(k.sites) == 1 && length(k.ops) == 1) ||
        throw(ArgumentError("$ctx must be a single-site charged operator"))
    return only(k.sites), only(k.ops), v
end

"""
    couple(a::TermSum, b::TermSum; to)
    a · b   (== couple(a, b; to = unit(I)))

Pairwise irrep coupling of two single-site ITO terms, fusing their operator charges to total `to`.
Singlet (`to == unit(I)`) applies the Cartesian normalization factor `-√dim(c)`; other targets use
unit Clebsch–Gordan. Multi-body / tree-structured coupling (`via`) is deferred.
"""
function couple(a::TermSum{I, S}, b::TermSum{I, S}; to, via = nothing) where {I, S}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    sa, opa, va = _single_charged_term(a, "couple: first operand")
    sb, opb, vb = _single_charged_term(b, "couple: second operand")
    sa == sb && throw(ArgumentError("couple: operators must act on distinct sites"))
    total = to::I

    # site order
    sites, ops = sa < sb ? (S[sa, sb], IrrepOperator{I}[opa, opb]) :
        (S[sb, sa], IrrepOperator{I}[opb, opa])

    # coupling channel must exist and be unique (pairwise, multiplicity-free)
    trees = collect(fusiontrees((ops[1].c, ops[2].c), total, (false, false)))
    @assert !isempty(trees) "charges $(ops[1].c),$(ops[2].c) do not fuse to $total"
    @assert length(trees) == 1 "expected a unique coupling channel; multi-channel (GenericFusion) coupling is deferred"

    coupfactor = total == unit(I) ? -sqrt(dim(ops[1].c)) : one(ComplexF64)
    coeff = ComplexF64(va) * ComplexF64(vb) * coupfactor

    d = Dictionary{TermKey{I, S}, ComplexF64}()
    insert!(d, TermKey{I, S}(sites, ops, total), coeff)
    return TermSum{I, S, ComplexF64}(d)
end

function couple(a::TermSum, b::TermSum, c::TermSum, rest::TermSum...; kwargs...)
    throw(ArgumentError("multi-body coupling (>2 operators) is deferred"))
end

# `·` / dot on term-sums == singlet coupling.
# NOTE: distinct from `LinearAlgebra.dot(::OperatorBasis, ::AbstractArray)` (basis projection).
function LinearAlgebra.dot(a::TermSum{I}, b::TermSum{I}) where {I}
    return couple(a, b; to = unit(I))
end

# Dense-oracle materialization
# ----------------------------
"""
    instantiate(ts::TermSum, sites::AbstractVector{<:ElementarySpace})

Materialize the term-sum into a TensorKit `TensorMap` over the lattice `sites` (the dense oracle),
summing each term. Supports identity (K=0), single-site field (K=1), and pairwise coupling (K=2).
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
    elseif K == 2
        return _embed_coupled(k.ops[1], k.sites[1], k.ops[2], k.sites[2], k.total, sites)
    else
        throw(ArgumentError("instantiation of $(K)-site terms is deferred"))
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

# pairwise-coupled two-site block (structural coupler; the reduced/-√dim factor is in the coeff)
function _embed_coupled(opa::IrrepOperator, pa, opb::IrrepOperator, pb, total, sites)
    Oa = instantiate(opa, sites[pa])               # V_pa ← V_pa ⊗ V_ca
    Ob = instantiate(opb, sites[pb])               # V_pb ← V_pb ⊗ V_cb
    Vca = domain(Oa)[2]
    Vcb = domain(Ob)[2]
    I = typeof(total)
    X = zeros(ComplexF64, Vca ⊗ Vcb ← Vect[I](total => 1))
    trees = collect(fusiontrees(X))
    isempty(trees) && throw(ArgumentError("charges do not fuse to $total"))
    for (f1, f2) in trees
        X[f1, f2] .= 1
    end
    @tensor Wab[o1 o2; i1 i2 t] := Oa[o1; i1 c1] * Ob[o2; i2 c2] * X[c1 c2; t]
    return _embed_block(Wab, pa, pb, sites)
end

# embed a two-site block on sites (pa, pb) into the full lattice, reordering to site order with
# the total-charge leg last in the domain.
function _embed_block(Wab, pa, pb, sites)
    N = length(sites)
    idle = [k for k in 1:N if k != pa && k != pb]
    full = isempty(idle) ? Wab : Wab ⊗ foldl(⊗, (id(sites[k]) for k in idle))

    cod_src = (pa, pb, idle...)
    p_cod = ntuple(j -> findfirst(==(j), cod_src), N)

    dom_src = (pa, pb, idle...)
    charge_global = N + 3
    site_global(j) = let pos = findfirst(==(j), dom_src)
        pos <= 2 ? N + pos : N + pos + 1
    end
    p_dom = (ntuple(j -> site_global(j), N)..., charge_global)
    return permute(full, (p_cod, p_dom))
end
