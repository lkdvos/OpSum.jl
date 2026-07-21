# Phase 2: symbolic layer over ITOs
# =================================
# This file wires the `IrrepOperator` alphabet (Phase 1) into the `LocalOp`/`GlobalOp`
# symbolic algebra, materializing symbolic operators into TensorKit `TensorMap`s (replacing
# the dense `kron` path) and adding the irrep-coupling surface (`couple`, `·`) plus named
# accessors (`spin`).
#
# Design notes:
# * The scalar identity is handled *structurally* as `c * id(V)` (see `instantiate` below),
#   never routed through a type-level `one(::Type{IrrepOperator})` (which cannot exist — the
#   identity depends on the physical space `V` and is generally a *sum* of trivial-charge ITOs,
#   not a single alphabet letter). Idle-site padding likewise uses TensorKit's `id(V)`.
# * On-site `Prod`/`Pow` of ITOs (fusion recoupling) are deferred to a later phase.
# * Coupling is pairwise only; multi-body / tree coupling errors with a Phase-3 pointer.

using TensorKit
using TensorKit: ElementarySpace, id, oneunit, unit, dim, sectors,
    fusiontrees, Vect, @tensor, domain, codomain, numin, numout, permute
import TensorKit: sectortype
using LinearAlgebra: LinearAlgebra
using .IrrepTensorOperators: IrrepOperator

export spin, scalarop, couple

# sector type of the ITO alphabet
sectortype(::Type{IrrepOperator{I}}) where {I} = I

# ITO algebra alias
const ITOLocalOp{T, I} = LocalOp{T, IrrepOperator{I}}
const ITOGlobalOp{T, I, S} = GlobalOp{T, IrrepOperator{I}, S}

# LocalOp instantiation over ITOs
# -------------------------------
"""
    instantiate(O::LocalOp{T,<:IrrepOperator}, V::ElementarySpace)

Materialize a symbolic local operator built over the ITO alphabet into a TensorKit `TensorMap`
on the physical space `V`.

* a **scalar** variant `c` maps to `c * id(V)` (structural identity — *not* `one(::Type)`);
* a bare `IrrepOperator` maps to its Phase-1 tensor `V ← V ⊗ Vect[I](c => 1)`;
* a `Sum` maps to the (scaled) sum of the term tensors.

On-site `Prod`/`Pow` (fusion recoupling) are deferred.
"""
function instantiate(O::LocalOp{T, A}, V::ElementarySpace) where {T, A <: IrrepOperator}
    o = variant(O)
    if o isa T
        return o * id(V)
    elseif o isa A
        return instantiate(o, V)
    elseif o isa Sum
        isempty(o.terms) && throw(ArgumentError("cannot instantiate an empty Sum without a space reference"))
        return sum(pairs(o.terms)) do (k, v)
            return v * instantiate(k, V)
        end
    elseif o isa Prod || o isa Pow
        throw(ArgumentError("on-site products/powers of ITOs (fusion recoupling) are deferred to a later phase"))
    else
        throw(ArgumentError("unsupported LocalOp variant for ITO instantiation: $(typeof(o))"))
    end
end

# Named accessors
# ---------------
"""
    spin(V::ElementarySpace)

The SU(2) rank-1 vector operator on a single-sector SU(2) space `V`, normalized so that
`spin(V)[i] · spin(V)[j]` densifies to the Cartesian `Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ`.

Concretely `spin(V) = √(s(s+1)(2s+1)) · IrrepOperator(SU2Irrep(1), 1)` where `s` is the spin of
the single irrep of `V` (the reduced matrix element `⟨s‖S‖s⟩`).
"""
function spin(V::ElementarySpace)
    sectortype(V) === SU2Irrep ||
        throw(ArgumentError("spin(V) requires an SU(2)-graded space, got sectortype $(sectortype(V))"))
    length(collect(sectors(V))) == 1 ||
        throw(ArgumentError("spin(V) requires a single SU(2) irrep sector"))
    j = only(sectors(V))
    s = j.j
    scale = sqrt(s * (s + 1) * (2s + 1))
    A = IrrepOperator{SU2Irrep}
    return scale * LocalOp{ComplexF64, A}(A(SU2Irrep(1), 1))
end

"""
    scalarop(c::Number, V::ElementarySpace)
    scalarop(c::Number, ::Type{I<:Sector})

A scalar (identity) local operator `c·𝟙` over the ITO alphabet with sector type `I`. Instantiates
*structurally* to `c * id(V)`; the identity is deliberately not represented as an alphabet letter.
"""
scalarop(c::Number, ::Type{I}) where {I} = LocalOp{ComplexF64, IrrepOperator{I}}(ComplexF64(c))
scalarop(c::Number, V::ElementarySpace) = scalarop(c, sectortype(V))

# GlobalOp instantiation over ITOs
# --------------------------------
"""
    instantiate(O::GlobalOp{T,<:IrrepOperator}, sites::AbstractVector{<:ElementarySpace})

Materialize a symbolic global operator into a TensorKit `TensorMap` over the physical `sites`
(one `ElementarySpace` per lattice site). Single-site `SiteOp`s (fields) are embedded with
structural `id(V)` on idle sites; if the field carries a nontrivial operator charge the result
retains a dangling charge leg (last domain leg). `Sum`s distribute. Multi-site placement of a
`SiteOp` (uncoupled product across sites) is not supported here — use [`couple`](@ref).
"""
function instantiate(
        O::GlobalOp{T, A, S}, sites::AbstractVector{<:ElementarySpace}
    ) where {T, A <: IrrepOperator, S}
    o = variant(O)
    if o isa SiteOp
        return _instantiate_ito_siteop(o, sites)
    elseif o isa Sum
        isempty(o.terms) && throw(ArgumentError("cannot instantiate an empty Sum without a space reference"))
        return sum(pairs(o.terms)) do (k, v)
            return v * instantiate(k, sites)
        end
    else
        throw(ArgumentError("unsupported GlobalOp variant for ITO instantiation: $(typeof(o)) (couplings go through `couple`)"))
    end
end

function _instantiate_ito_siteop(o, sites::AbstractVector{<:ElementarySpace})
    @assert issorted(o.sites) && allunique(o.sites) "Sites must be sorted and unique."
    N = length(sites)

    # identity operator (no site indices): structural identity on every site
    isempty(o.sites) && return foldl(⊗, (id(sites[k]) for k in 1:N))

    length(o.sites) == 1 ||
        throw(ArgumentError("multi-site SiteOp embedding of ITOs is not supported; express couplings with `couple`"))

    p = only(o.sites)
    loc = instantiate(o.op, sites[p])              # V_p ← V_p [⊗ V_c]
    has_charge = numin(loc) == 2

    # tensor product in site order
    full = foldl(⊗, (k == p ? loc : id(sites[k]) for k in 1:N))
    has_charge || return full

    # permute the (site-p) charge leg to the last domain slot
    # full domain leg order: V_1..V_{p-1}, V_p, V_c, V_{p+1}..V_N  → move V_c to the end
    cod = ntuple(identity, N)
    charge_global = N + (p + 1)
    dom_wo_charge = (ntuple(m -> N + m, p)..., ntuple(m -> N + p + 1 + m, N - p)...)
    dom = (dom_wo_charge..., charge_global)
    return permute(full, (cod, dom))
end

# Coupling (irrep-coupled products)
# ---------------------------------
"""
    CoupledOp(a, b, to)

A pairwise irrep-coupled product of two single-site ITO `GlobalOp`s `a`, `b`, fusing their
operator charge legs to a total charge `to::Sector`. Constructed via [`couple`](@ref); it lives
outside the automaton `GlobalOp` sum-type on purpose (the fusion structure only reaches the trie
in a later phase).
"""
struct CoupledOp{Ga, Gb, C}
    a::Ga
    b::Gb
    to::C
end

"""
    couple(a::GlobalOp, b::GlobalOp; to)

Couple two single-site ITO operators `a`, `b` by fusing their operator charge legs to the total
charge `to`. Returns a [`CoupledOp`](@ref) that materializes (via `instantiate`) to a global
`TensorMap`.

Coupling is **pairwise only**. Coupling to the singlet (`to = unit(I)`, also spelled `a · b`)
applies the Cartesian normalization factor `-√dim(c)` so that `spin(V)[i] · spin(V)[j]` densifies
to `Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ`; other targets use the bare Clebsch–Gordan structure.
"""
function couple(a::GlobalOp, b::GlobalOp; to, via = nothing)
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred to Phase 3"))
    variant(a) isa SiteOp && length(variant(a).sites) == 1 ||
        throw(ArgumentError("couple: first operand must be a single-site operator"))
    variant(b) isa SiteOp && length(variant(b).sites) == 1 ||
        throw(ArgumentError("couple: second operand must be a single-site operator"))
    return CoupledOp(a, b, to)
end

# multi-body coupling: explicit Phase-3 stub
function couple(a::GlobalOp, b::GlobalOp, c::GlobalOp, rest::GlobalOp...; kwargs...)
    throw(ArgumentError("multi-body coupling (>2 operators) is deferred to Phase 3"))
end

# `·` / dot on global ops == singlet coupling.
# NOTE: distinct from `LinearAlgebra.dot(::OperatorBasis, ::AbstractArray)` (basis projection).
function LinearAlgebra.dot(a::GlobalOp{Ta, A}, b::GlobalOp{Tb, B}) where {Ta, Tb, A <: IrrepOperator, B <: IrrepOperator}
    I = sectortype(A)
    sectortype(B) === I || throw(ArgumentError("dot: incompatible sector types"))
    return couple(a, b; to = unit(I))
end

function instantiate(cop::CoupledOp, sites::AbstractVector{<:ElementarySpace})
    sa = variant(cop.a)
    sb = variant(cop.b)
    pa = only(sa.sites)
    pb = only(sb.sites)
    pa == pb && throw(ArgumentError("couple: the two operators must act on distinct sites"))

    Oa = instantiate(sa.op, sites[pa])   # V_pa ← V_pa ⊗ V_ca
    Ob = instantiate(sb.op, sites[pb])   # V_pb ← V_pb ⊗ V_cb
    numin(Oa) == 2 && numin(Ob) == 2 ||
        throw(ArgumentError("couple: both operands must carry an operator charge leg (not scalars)"))

    Vca = domain(Oa)[2]
    Vcb = domain(Ob)[2]
    ca = only(sectors(Vca))
    cb = only(sectors(Vcb))
    I = typeof(ca)
    to = cop.to::I

    # coupler X : V_ca ⊗ V_cb ← Vect[I](to => 1)
    Vto = Vect[I](to => 1)
    X = zeros(ComplexF64, Vca ⊗ Vcb ← Vto)
    ftrees = collect(fusiontrees(X))
    isempty(ftrees) &&
        throw(ArgumentError("charges $ca and $cb do not fuse to $to"))
    for (f1, f2) in ftrees
        X[f1, f2] .= 1
    end
    # Cartesian singlet normalization
    if to == unit(I)
        X = (-sqrt(dim(ca))) * X
    end

    # two-site coupled block: (V_pa ⊗ V_pb) ← (V_pa ⊗ V_pb) ⊗ Vto
    @tensor Wab[o1 o2; i1 i2 t] := Oa[o1; i1 c1] * Ob[o2; i2 c2] * X[c1 c2; t]

    return _embed_block(Wab, pa, pb, sites)
end

# Embed the two-site coupled block on sites (pa, pb) into the full lattice (identities on idle
# sites), reordering legs to site order with the (total-charge) leg last in the domain.
function _embed_block(Wab, pa::Integer, pb::Integer, sites::AbstractVector{<:ElementarySpace})
    N = length(sites)
    idle = [k for k in 1:N if k != pa && k != pb]

    full = isempty(idle) ? Wab : Wab ⊗ foldl(⊗, (id(sites[k]) for k in idle))

    # codomain: full legs correspond to sites [pa, pb, idle...]; permute into site order 1:N
    cod_src = (pa, pb, idle...)
    p_cod = ntuple(j -> findfirst(==(j), cod_src), N)

    # domain: full legs correspond to [pa, pb, charge, idle...]; site legs in order, charge last
    dom_src = (pa, pb, idle...)  # site legs only (charge handled separately)
    # global domain index layout of `full`: N+1 -> pa, N+2 -> pb, N+3 -> charge, N+4.. -> idle
    charge_global = N + 3
    site_global(j) = begin
        pos = findfirst(==(j), dom_src)
        pos <= 2 ? N + pos : N + pos + 1  # skip the charge leg sitting at position 3
    end
    p_dom = (ntuple(j -> site_global(j), N)..., charge_global)
    return permute(full, (p_cod, p_dom))
end
