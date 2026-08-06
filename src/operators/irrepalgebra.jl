# Term algebra over ITOs: `Term` (one term) → `Terms` (a bag) → `TermSum` (a bag + its lattice).
# Site labels are lattice indices; on-site products and `GenericFusion` multiplicity are deferred.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, fusiontrees, unit, dim, id, Nsymbol,
    Vect, domain, permute, sectors, fuse, FusionStyle, UniqueFusion
import TensorKit: sectortype
using LinearAlgebra: LinearAlgebra
using .IrrepTensorOperators: IrrepOperator

# sector type of the ITO alphabet
sectortype(::Type{IrrepOperator{I}}) where {I} = I

"""
    instantiate(O::SiteOperator, V::ElementarySpace)

Materialize a single-site operator over the ITO alphabet as the coefficient-weighted sum of its
letters' tensors. The pass-through letter instantiates to `id(V)`, so a scalar `c·𝟙` comes out as
`c * id(V)` structurally.
"""
function instantiate(O::SiteOperator{I}, V::ElementarySpace) where {I}
    isempty(O) && throw(ArgumentError("cannot instantiate an empty SiteOperator without a space"))
    return sum(((l, c),) -> c * instantiate(l, V), pairs(O))
end

const _SPIN_CACHE = Dict{ElementarySpace, Any}()

"""
    spin(V::ElementarySpace)

The SU(2) rank-1 vector operator on a single-sector SU(2) space `V`, normalized so `spin(V)[i] ·
spin(V)[j]` densifies to the Cartesian `Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ`. Concretely
`spin(V) = √(s(s+1)(2s+1)) · IrrepOperator(SU2Irrep(1), 1)` (the reduced matrix element).

Memoised per space, so calling it inside a term loop is a dictionary lookup. See also
[`spin_ops`](@ref) for the U(1)-graded `(Sp, Sm, Sz)` form.
"""
function spin(V::ElementarySpace)
    return _cached(_SPIN_CACHE, V) do
        _spin(V)
    end
end

function _spin(V::ElementarySpace)
    sectortype(V) === SU2Irrep ||
        throw(ArgumentError("spin(V) requires an SU(2)-graded space, got $(sectortype(V))"))
    length(collect(sectors(V))) == 1 ||
        throw(ArgumentError("spin(V) requires a single SU(2) irrep sector"))
    s = only(sectors(V)).j
    scale = sqrt(s * (s + 1) * (2s + 1))
    return scale * SiteOperator(IrrepOperator{SU2Irrep}(SU2Irrep(1), 1))
end

"""
    scalarop(c::Number, V::ElementarySpace)
    scalarop(c::Number, ::Type{I<:Sector})

A scalar (identity) local operator `c·𝟙` over the ITO alphabet; instantiates structurally to
`c * id(V)` (the identity is not an alphabet letter).
"""
scalarop(c::Number, ::Type{I}) where {I} = SiteOperator{I}(ComplexF64(c))
scalarop(c::Number, V::ElementarySpace) = scalarop(c, sectortype(V))

# The 4-arg `FusionTree` form throughout: the 3-arg one is an abelian-only shortcut that throws for
# `MultipleFusion`.
_idtree(::Type{I}) where {I <: Sector} = FusionTree{I}((), unit(I), (), ())

# Inverse of `bondcharges`/`vertexlabels`, and the reason a `Term` need not store a tree.
function _tree_from_bonds(charges::AbstractVector{I}, bonds::AbstractVector{I}, verts::AbstractVector{Int}) where {I <: Sector}
    K = length(charges)
    K == 0 && return _idtree(I)
    unc = ntuple(i -> charges[i], K)
    isd = ntuple(_ -> false, K)
    inner = ntuple(i -> bonds[i + 1], max(0, K - 2))   # innerlines = bonds[2 : K-1]
    vtx = ntuple(i -> verts[i + 1], max(0, K - 1))     # vertices   = verts[2 : K]
    return FusionTree{I}(unc, bonds[K], isd, inner, vtx)
end

"""
    Term{I<:Sector}

A single term: `sites` (ascending, unique), one [`ITOKey`](@ref) each — the letter plus the running
bond charge and vertex label at that position — and a reduced `coeff`.

`arity(t)` is 0 for the identity, 1 for a field, ≥ 2 for left-nested (caterpillar) coupling, whose
tree is *derived* from the running bond charges rather than stored ([`tree`](@ref), irrepkey.jl).
Multiplicity-free fusion only; `GenericFusion` is deferred.

Never mutated, so terms may share the vectors they were built from. `==`/`hash` ignore the
coefficient, which is what makes summing coincident terms well defined.
"""
struct Term{I <: Sector}
    sites::Vector{Int}
    keys::Vector{ITOKey{I}}
    coeff::ComplexF64
    function Term{I}(sites::Vector{Int}, keys::Vector{ITOKey{I}}, coeff::ComplexF64) where {I}
        @assert length(sites) == length(keys) "sites/keys arity mismatch"
        @assert issorted(sites) && allunique(sites) "a term's sites must be ascending and unique"
        @assert all(k -> isone(k.vertex), keys) "multiplicity-free fusion only (all vertices == 1)"
        return new{I}(sites, keys, coeff)
    end
end
Term(sites::Vector{Int}, keys::Vector{ITOKey{I}}, coeff::Number) where {I} =
    Term{I}(sites, keys, ComplexF64(coeff))

sectortype(::Type{Term{I}}) where {I} = I

"""
    arity(t::Term) -> Int

The number of active (charged) sites of `t`.
"""
arity(t::Term) = length(t.sites)

"""
    ops(t::Term) -> Vector{IrrepOperator}

The alphabet letters of `t`, one per active site, in site order.
"""
ops(t::Term{I}) where {I} = IrrepOperator{I}[k.op for k in t.keys]

"""
    total(t::Term) -> Sector

The total charge `t` couples to: the running bond charge after its last active site, or the unit
sector for the identity term.
"""
total(t::Term{I}) where {I} = isempty(t.keys) ? unit(I) : last(t.keys).bond

"""
    tree(t::Term) -> FusionTree

The left-nested (caterpillar) coupling tree of `t`, rebuilt from its stored running bond charges and
vertex labels. Derived, not stored: for a caterpillar the two are the same information (irrepkey.jl).
"""
function tree(t::Term{I}) where {I}
    K = arity(t)
    K == 0 && return _idtree(I)
    charges = I[k.op.c for k in t.keys]
    bonds = I[k.bond for k in t.keys]
    verts = Int[k.vertex for k in t.keys]
    return _tree_from_bonds(charges, bonds, verts)
end

# Coefficient excluded: identity is "same term", so coincident terms can be summed.
Base.:(==)(x::Term{I}, y::Term{I}) where {I} = x.sites == y.sites && x.keys == y.keys
Base.hash(t::Term, h::UInt) = hash(t.keys, hash(t.sites, hash(:ITOTerm, h)))

# Total order for the sort-and-merge normal form; the keys distinguish letters *and* couplings.
function Base.isless(x::Term{I}, y::Term{I}) where {I}
    x.sites == y.sites || return isless(x.sites, y.sites)
    return isless(x.keys, y.keys)
end

VectorInterface.scale(t::Term{I}, α::Number) where {I} =
    Term{I}(t.sites, t.keys, t.coeff * ComplexF64(α))
Base.:*(α::Number, t::Term) = scale(t, α)
Base.:*(t::Term, α::Number) = scale(t, α)
Base.:/(t::Term, α::Number) = scale(t, inv(α))
Base.:-(t::Term) = scale(t, -1)

function Base.show(io::IO, t::Term)
    return print(io, "Term(sites=", t.sites, ", ops=", ops(t), ", total=", total(t), ", coeff=", t.coeff, ")")
end

"""
    Terms{I<:Sector}

A bag of [`Term`](@ref)s with no lattice: what `A[i]`, [`couple`](@ref), `dot` and
[`project`](@ref) return, and what `+`, `-`, `*` and `/` combine. Nothing is deduplicated here — the
normal form comes later, on the [`TermSum`](@ref). [`opsum`](@ref) makes it compressible.
"""
struct Terms{I <: Sector}
    terms::Vector{Term{I}}
end
Terms{I}() where {I <: Sector} = Terms{I}(Term{I}[])
Terms(t::Term{I}) where {I} = Terms{I}(Term{I}[t])

sectortype(::Type{Terms{I}}) where {I} = I

Base.length(ts::Terms) = length(ts.terms)
Base.isempty(ts::Terms) = isempty(ts.terms)
Base.iterate(ts::Terms, args...) = iterate(ts.terms, args...)
Base.eltype(::Type{Terms{I}}) where {I} = Term{I}
Base.getindex(ts::Terms, i::Integer) = ts.terms[i]
Base.firstindex(::Terms) = 1
Base.lastindex(ts::Terms) = length(ts.terms)

Base.:+(a::Terms{I}, b::Terms{I}) where {I} = Terms{I}(vcat(a.terms, b.terms))
VectorInterface.scale(ts::Terms{I}, α::Number) where {I} =
    Terms{I}(Term{I}[scale(t, α) for t in ts.terms])
Base.:*(α::Number, ts::Terms) = scale(ts, α)
Base.:*(ts::Terms, α::Number) = scale(ts, α)
Base.:/(ts::Terms, α::Number) = scale(ts, inv(α))
Base.:-(ts::Terms) = scale(ts, -1)
Base.:-(a::Terms{I}, b::Terms{I}) where {I} = a + (-b)

Base.one(::Terms{I}) where {I} = Terms{I}(Term{I}[Term{I}(Int[], ITOKey{I}[], ComplexF64(1))])

# On *copies*: bags are unnormalised, and the operands must not be reordered under the caller.
Base.isapprox(a::Terms{I}, b::Terms{I}; kwargs...) where {I} =
    _termsapprox(_canonicalize!(copy(a.terms)), _canonicalize!(copy(b.terms)); kwargs...)
Base.:(==)(a::Terms{I}, b::Terms{I}) where {I} =
    _termsequal(_canonicalize!(copy(a.terms)), _canonicalize!(copy(b.terms)))

function Base.show(io::IO, ts::Terms)
    print(io, "Terms(")
    join(io, ("$(t.coeff) * $(_termbody(t))" for t in ts.terms), " + ")
    return print(io, ")")
end
_termbody(t::Term) = string("[sites=", t.sites, ", ops=", ops(t), ", total=", total(t), "]")

# `charge => number of letters of that charge on V`
function _alphabet(V::ElementarySpace, ::Type{I}) where {I <: Sector}
    W = fuse(V ⊗ V')
    return Dict{I, Int}(c => dim(W, c) for c in sectors(W))
end

_tolattice(sites::Vector{<:ElementarySpace}) = sites
function _tolattice(sites)
    lat = collect(sites)
    eltype(lat) <: ElementarySpace || throw(
        ArgumentError("a lattice must be a vector of `ElementarySpace`s, got $(eltype(lat))")
    )
    return lat
end

# The only place operators and the spaces they are compressed with are confronted; without it a
# `sites` from a different model silently gives a wrong MPO. `Θ(Σ arity)`, not `Θ(N)`, so that
# `H + t` stays proportional to `t`.
function _checkterms(terms::AbstractVector{Term{I}}, lat::Vector{<:ElementarySpace}) where {I}
    N = length(lat)
    isempty(terms) && return nothing
    seen = Dict{eltype(lat), Dict{I, Int}}()
    for t in terms
        for (s, key) in zip(t.sites, t.keys)
            1 <= s <= N || throw(
                ArgumentError("a term acts on site $s, outside the lattice `1:$N`")
            )
            V = lat[s]
            info = get!(() -> _alphabet(V, I), seen, V)
            op = key.op
            nmax = get(info, op.c, 0)
            1 <= op.n <= nmax || throw(
                ArgumentError(
                    "letter $op does not exist on site $s (space $V): that space carries " *
                        "$nmax operator(s) of charge $(op.c)"
                )
            )
        end
    end
    return nothing
end

"""
    TermSum{I<:Sector}

The compressible ITO operator: a list of [`Term`](@ref)s plus the `lattice` they live on. The lattice
is **mandatory** — `instantiate` and `irrep_mpo_tensors` need the space of every site, idle ones
included, which no term can know — so a `TermSum` comes from [`opsum`](@ref), not from placement, and
[`irrep_mpo`](@ref) / [`instantiate`](@ref) / [`jordan_mpo_tensors`](@ref) need no `sites`.

Terms are appended as given, so the list is a bag until [`canonicalize!`](@ref) normalises it in place.
`length(H)`, iteration and `≈` therefore report the *canonical* operator, never the append count
(`nterms_raw(H)` is that, for tests).
"""
struct TermSum{I <: Sector}
    lattice::Vector{<:ElementarySpace}
    terms::Vector{Term{I}}
end

sectortype(::Type{TermSum{I}}) where {I} = I

"""
    lattice(H::TermSum) -> Vector{<:ElementarySpace}

The physical spaces `H` is defined on, one per site. Always present.
"""
lattice(H::TermSum) = H.lattice

"""
    opsum(sites, terms...) -> TermSum

Bind terms to the lattice `sites` (one physical space per site), in **one pass** over each argument,
so building an M-term Hamiltonian costs `Θ(M)`:

```julia
H = opsum(fill(V, N), (J * dot(S[i], S[i + 1]) for i in 1:(N - 1)))
```

Each argument may be a [`Term`](@ref), a [`Terms`](@ref) bag, another `TermSum` on the same lattice,
or any iterable of those, nested arbitrarily. Every letter is checked against the space of the site it
acts on and every site against `1:length(sites)` — the only place operators and spaces are confronted.

`opsum(sites)` is the empty operator; add to it with `+` or [`append!`](@ref).
"""
function opsum(sites, args...)
    lat = _tolattice(sites)
    # The sector type comes from the lattice, never from the terms: it is always known there, and an
    # argument may be a one-shot generator that must not be iterated twice to be sniffed.
    I = _lattice_sectortype(lat)
    out = Term{I}[]
    for a in args
        _collect_terms!(out, a, I)
    end
    _checkterms(out, lat)
    return TermSum{I}(lat, out)
end

_lattice_sectortype(lat::Vector{<:ElementarySpace}) =
    isempty(lat) ? Trivial : sectortype(first(lat))

_collect_terms!(out::Vector{Term{I}}, t::Term{I}, ::Type{I}) where {I} = push!(out, t)
_collect_terms!(out::Vector{Term{I}}, ts::Terms{I}, ::Type{I}) where {I} =
    append!(out, ts.terms)
_collect_terms!(out::Vector{Term{I}}, H::TermSum{I}, ::Type{I}) where {I} =
    append!(out, H.terms)
function _collect_terms!(out::Vector{Term{I}}, itr, ::Type{I}) where {I}
    for a in itr
        _collect_terms!(out, a, I)
    end
    return out
end
# A term over a different symmetry than the lattice is a mistake worth naming.
_collect_terms!(::Vector{Term{I}}, ::Term{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_collect_terms!(::Vector{Term{I}}, ::Terms{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_collect_terms!(::Vector{Term{I}}, ::TermSum{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_wrongsector(I, J) = throw(
    ArgumentError("cannot place an operator over $J on a lattice of sector type $I")
)

TermSum(sites) = opsum(sites)
TermSum(sites, args...) = opsum(sites, args...)

"""
    append!(H::TermSum, terms...) -> H

Append terms to `H` **in place**, in one pass — the linear way to accumulate into an existing
operator. Same argument forms as [`opsum`](@ref).
"""
function Base.append!(H::TermSum{I}, args...) where {I}
    isempty(args) && return H
    added = Term{I}[]
    for a in args
        _collect_terms!(added, a, I)
    end
    _checkterms(added, H.lattice)
    append!(H.terms, added)
    return H
end

# Copies, so older values stay valid — but folding it over M terms is quadratic; use `opsum`.
_addterms(H::TermSum{I}, args...) where {I} =
    (out = TermSum{I}(H.lattice, copy(H.terms)); append!(out, args...); out)
Base.:+(H::TermSum, t::Term) = _addterms(H, t)
Base.:+(H::TermSum, ts::Terms) = _addterms(H, ts)
Base.:+(t::Term, H::TermSum) = _addterms(H, t)
Base.:+(ts::Terms, H::TermSum) = _addterms(H, ts)
function Base.:+(a::TermSum{I}, b::TermSum{I}) where {I}
    a.lattice == b.lattice ||
        throw(ArgumentError("cannot add operators defined on different lattices"))
    return _addterms(a, b)
end
# negating first, then `+`, keeps the lattice check
Base.:-(H::TermSum, x) = H + (-x)

VectorInterface.scale(H::TermSum{I}, α::Number) where {I} =
    TermSum{I}(H.lattice, Term{I}[scale(t, α) for t in H.terms])
Base.:*(α::Number, H::TermSum) = scale(H, α)
Base.:*(H::TermSum, α::Number) = scale(H, α)
Base.:/(H::TermSum, α::Number) = scale(H, inv(α))
Base.:-(H::TermSum) = scale(H, -1)

"""
    nterms_raw(H::TermSum) -> Int

The number of *appended* terms, before coincident ones are summed. For tests; `length(H)` is the
number of terms the operator actually has.
"""
nterms_raw(H::TermSum) = length(H.terms)

Base.length(H::TermSum) = length(canonicalize!(H).terms)
Base.isempty(H::TermSum) = isempty(canonicalize!(H).terms)
Base.iterate(H::TermSum, args...) = iterate(canonicalize!(H).terms, args...)
Base.eltype(::Type{TermSum{I}}) where {I} = Term{I}
Base.getindex(H::TermSum, i::Integer) = canonicalize!(H).terms[i]
Base.firstindex(::TermSum) = 1
Base.lastindex(H::TermSum) = length(H)

"""
    isapprox(a::TermSum, b::TermSum; kwargs...)
    a ≈ b

Whether two operators carry the same terms with matching coefficients: the canonical term sets must
be **equal** (a dropped term is never "approximately" absent) and the coefficients `≈`. Lattices are
not compared, so an operator reconstructed by [`mpo_terms`](@ref) — which knows the bonds but not the
spaces — compares equal to the one it came from. This is the faithfulness check.
"""
Base.isapprox(a::TermSum{I}, b::TermSum{I}; kwargs...) where {I} =
    _termsapprox(canonicalize!(a).terms, canonicalize!(b).terms; kwargs...)
Base.:(==)(a::TermSum{I}, b::TermSum{I}) where {I} =
    _termsequal(canonicalize!(a).terms, canonicalize!(b).terms)

function Base.show(io::IO, H::TermSum)
    canonicalize!(H)
    print(io, "TermSum(")
    join(io, ("$(t.coeff) * $(_termbody(t))" for t in H.terms), " + ")
    return print(io, ")")
end

function Base.getindex(O::SiteOperator{I}, ind::Integer, inds::Integer...) where {I}
    isempty(inds) ||
        throw(ArgumentError("multi-site placement of an on-site operator is not supported; use `couple`"))
    site = Int(ind)
    site >= 1 || throw(ArgumentError("site index must be ≥ 1, got $site"))
    out = Term{I}[]
    sizehint!(out, length(O))
    for (letter, coeff) in pairs(O)
        # Pass-through is the bare identity, so it places as a K=0 term. A real trivial-charge
        # letter (`n ≥ 1`) is a different thing and does place.
        if ispassthrough(letter)
            push!(out, Term{I}(Int[], ITOKey{I}[], ComplexF64(coeff)))
        else
            push!(
                out,
                Term{I}([site], ITOKey{I}[ITOKey{I}(letter, letter.c, 1)], ComplexF64(coeff))
            )
        end
    end
    return Terms{I}(out)
end

# Second operand of `dot`: a single-site charged term; returns `(site, op, coeff)`.
function _single_site_term(b::Terms{I}, ctx) where {I}
    length(b) == 1 || throw(ArgumentError("$ctx must be a single-term operator"))
    t = only(b.terms)
    arity(t) == 1 || throw(ArgumentError("$ctx must be a single-site charged operator"))
    return only(t.sites), only(t.keys).op, t.coeff
end

# Extend every term of `a` by every single-site term of `b`, fusing to `target(running total, b's
# letter)`; unreachable pairs are dropped, so the result may be empty. Extending a caterpillar leaves
# the earlier running charges alone, so this is an append plus one `Nsymbol` — no tree is built.
function _couple_terms(a::Terms{I}, b::Terms{I}, target) where {I}
    out = Term{I}[]
    sizehint!(out, length(a) * length(b))
    for ta in a.terms
        na = arity(ta)
        na >= 1 || throw(
            ArgumentError("couple: every term of the first operand must carry at least one charged operator")
        )
        run = last(ta.keys).bond
        lastsite = last(ta.sites)
        for tb in b.terms
            arity(tb) == 1 || throw(
                ArgumentError("couple: every term of the second operand must be a single-site charged operator")
            )
            sb = only(tb.sites)
            opb = only(tb.keys).op
            sb == lastsite && throw(ArgumentError("couple: operators must act on distinct sites"))
            lastsite < sb || throw(
                ArgumentError(
                    "couple: the second operand must act to the right of the first " *
                        "(left-nested caterpillar; out-of-order coupling is deferred)"
                )
            )

            tot = target(run, opb)::I
            nsym = Nsymbol(run, opb.c, tot)
            iszero(nsym) && continue    # this pair of charges cannot fuse to `tot`: drop it
            @assert isone(nsym) "expected a unique coupling channel; multi-channel (GenericFusion) coupling is deferred"

            push!(
                out,
                Term{I}(
                    push!(copy(ta.sites), sb),
                    push!(copy(ta.keys), ITOKey{I}(opb, tot, 1)),
                    ta.coeff * tb.coeff
                )
            )
        end
    end
    return Terms{I}(out)
end

"""
    couple(a::Terms, b::Terms; to = unit(I))
    couple(a::Terms, b::Terms, cs::Terms...; to = unit(I))   # abelian only

Left-nested (caterpillar) irrep coupling: extend the composite `a` (any K ≥ 1 sites, with its running
coupling charges) by one single-site operator `b`, fusing the running total of `a` with `b`'s charge
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
function couple(a::Terms{I}, b::Terms{I}; to = unit(I), via = nothing) where {I}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    isempty(a) && throw(ArgumentError("couple: first operand has no terms"))
    isempty(b) && throw(ArgumentError("couple: second operand has no terms"))
    tot = to::I

    out = _couple_terms(a, b, (_, _) -> tot)
    isempty(out) &&
        throw(ArgumentError("couple: no pair of terms of the operands fuses to $tot"))
    return out
end

# Abelian only: unique fusion forces every intermediate, so fold left and constrain only the total.
function couple(
        a::Terms{I}, b::Terms{I}, c::Terms{I}, rest::Terms{I}...;
        to = unit(I), via = nothing
    ) where {I}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    FusionStyle(I) isa UniqueFusion || throw(
        ArgumentError(
            "couple: variadic coupling needs an abelian symmetry (unique fusion), but $I has " *
                "$(FusionStyle(I)) — the intermediate channels are a real choice there. Nest " *
                "`couple` to name each one, e.g. couple(couple(a, b; to = b₂), c; to = t)"
        )
    )
    isempty(a) && throw(ArgumentError("couple: first operand has no terms"))
    tot = to::I

    others = (b, c, rest...)
    acc = a
    for (i, nxt) in enumerate(others)
        isempty(nxt) &&
            throw(ArgumentError("couple: operand $(i + 1) has no terms"))
        # intermediates are forced; only the last step has to land on `to`
        acc = if i == length(others)
            _couple_terms(acc, nxt, (_, _) -> tot)
        else
            _couple_terms(acc, nxt, (ta, opb) -> only(ta ⊗ opb.c))
        end
        isempty(acc) && throw(
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
    dot(a::Terms, b::Terms)
    a · b

The Cartesian two-body scalar product of two single-site ITO operators: singlet coupling
`couple(a, b; to = unit(I))` times the Cartesian factor `-√dim(c)` (the identity
`Sᵢ·Sⱼ = -√3 [S⊗S]⁽⁰⁾` with `-√3 = -√dim(spin-1)`). Order-independent in the two sites: the sites
are sorted before coupling, and the Cartesian factor is read off the operator that ends up on the
left. (The two charges must fuse to the unit sector, i.e. be each other's dual, and dual sectors have
equal quantum dimension — so `dot(a, b) == dot(b, a)` exactly.)

Unlike [`couple`](@ref) this does not distribute over composite operands: the Cartesian factor
`-√dim(c)` is per-letter, so it has no meaning for an operator mixing several charges. Use `couple`
for those.
"""
function LinearAlgebra.dot(a::Terms{I}, b::Terms{I}) where {I}
    sa, opa, _ = _single_site_term(a, "·: first operand")
    sb, opb, _ = _single_site_term(b, "·: second operand")
    sa == sb && throw(ArgumentError("·: operators must act on distinct sites"))
    iszero(Nsymbol(opa.c, opb.c, unit(I))) && throw(
        ArgumentError(
            "·: the two operator charges $(opa.c) and $(opb.c) do not fuse to the unit sector, so " *
                "there is no scalar product; use `couple(a, b; to = …)` for a charged coupling"
        )
    )
    # Sort first, *then* read the Cartesian factor off the left operand, so that the factor cannot
    # depend on the order the caller wrote the operands in.
    lo, hi, oplo = sa < sb ? (a, b, opa) : (b, a, opb)
    return scale(couple(lo, hi; to = unit(I)), -sqrt(dim(oplo.c)))
end

# Dense-oracle materialization
# ----------------------------
"""
    instantiate(H::TermSum)
    instantiate(ts::Terms, sites::AbstractVector{<:ElementarySpace})

Materialize the operator into a TensorKit `TensorMap` over its lattice (the dense oracle), summing
each term. Supports identity (K=0), single-site field (K=1), and left-nested (caterpillar) coupling
of any K ≥ 2 sites.
"""
function instantiate(H::TermSum)
    isempty(H) && throw(ArgumentError("cannot instantiate an empty TermSum"))
    sites = H.lattice
    length(sites) == 0 && throw(ArgumentError("cannot instantiate over an empty lattice"))
    return sum(t -> t.coeff * _instantiate_term(t, sites), H.terms)
end
instantiate(ts::Terms, sites::AbstractVector{<:ElementarySpace}) = instantiate(opsum(sites, ts))

function _instantiate_term(t::Term, sites)
    K = arity(t)
    K == 0 && return foldl(⊗, (id(V) for V in sites))
    K == 1 && return _embed_field(only(t.keys).op, only(t.sites), sites)
    return _embed_caterpillar(ops(t), t.sites, tree(t), sites)
end

# The candidate basis element for a `(letters, tree)` combination on the local lattice `1:K` — the
# same forward map, entered without building a `Term`, which is what `project` takes inner products
# against.
function _instantiate_basis(ops, tree, sites)
    K = length(ops)
    K == 0 && return foldl(⊗, (id(V) for V in sites))
    K == 1 && return _embed_field(only(ops), 1, sites)
    return _embed_caterpillar(ops, 1:K, tree, sites)
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

# Caterpillar K-site block; the coupler `X` selects the specific channel `tree`. `permute` +
# composition rather than `@tensor`, so it generalises to any K.
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
