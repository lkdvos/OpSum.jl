# Term algebra over ITOs: `Term` (one term) → `Terms` (a bag, the operator). Latticeless: the
# physical spaces are supplied where the MPO is formed, not here.
# Site labels are lattice indices; on-site products and `GenericFusion` multiplicity are deferred.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, fusiontrees, unit, dim, id, Nsymbol,
    Rsymbol, Vect, domain, permute, sectors, fuse, FusionStyle, UniqueFusion, BraidingStyle,
    SymmetricBraiding, insertrightunit
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

The compressible ITO operator: a bag of [`Term`](@ref)s, with **no lattice**. What `A[i]`,
[`couple`](@ref), `dot` and [`project`](@ref) return, what `+`, `-`, `*` and `/` combine, and what
[`opsum`](@ref) accumulates in one pass.

Nothing in the term algebra, and nothing in the MPO sweep, needs a physical space — the sweep needs
only the site *count* — so the lattice is supplied where the MPO is formed:
`irrep_mpo(h, lat)`, `instantiate(h, lat)`, `islossless(h, lat)`, `adjoint(h, lat)`.

Terms are appended as given, so the bag is unnormalised until [`canonicalize!`](@ref) puts it in
normal form in place. `length(ts)`, iteration, indexing and `≈` all go through that, so they report
the *canonical* operator and never the append count (`nterms_raw(ts)` is that, for tests).
"""
struct Terms{I <: Sector}
    terms::Vector{Term{I}}
end
Terms{I}() where {I <: Sector} = Terms{I}(Term{I}[])
Terms(terms::Vector{Term{I}}) where {I} = Terms{I}(terms)
Terms(t::Term{I}) where {I} = Terms{I}(Term{I}[t])

sectortype(::Type{Terms{I}}) where {I} = I

"""
    nterms_raw(ts::Terms) -> Int

The number of *appended* terms, before coincident ones are summed. For tests; `length(ts)` is the
number of terms the operator actually has.
"""
nterms_raw(ts::Terms) = length(ts.terms)

# Everything that observes the term set normalises first, so an append count can never be mistaken
# for the number of terms the operator has.
Base.length(ts::Terms) = length(canonicalize!(ts).terms)
Base.isempty(ts::Terms) = isempty(canonicalize!(ts).terms)
Base.iterate(ts::Terms, args...) = iterate(canonicalize!(ts).terms, args...)
Base.eltype(::Type{Terms{I}}) where {I} = Term{I}
Base.getindex(ts::Terms, i::Integer) = canonicalize!(ts).terms[i]
Base.firstindex(::Terms) = 1
Base.lastindex(ts::Terms) = length(ts)

# Copies, so older values stay valid — but folding it over M terms is quadratic; use `opsum`.
Base.:+(a::Terms{I}, b::Terms{I}) where {I} = Terms{I}(vcat(a.terms, b.terms))
Base.:+(a::Terms{I}, t::Term{I}) where {I} = Terms{I}(vcat(a.terms, [t]))
Base.:+(t::Term{I}, a::Terms{I}) where {I} = Terms{I}(vcat([t], a.terms))
VectorInterface.scale(ts::Terms{I}, α::Number) where {I} =
    Terms{I}(Term{I}[scale(t, α) for t in ts.terms])
Base.:*(α::Number, ts::Terms) = scale(ts, α)
Base.:*(ts::Terms, α::Number) = scale(ts, α)
Base.:/(ts::Terms, α::Number) = scale(ts, inv(α))
Base.:-(ts::Terms) = scale(ts, -1)
Base.:-(a::Terms{I}, b::Terms{I}) where {I} = a + (-b)
Base.:-(a::Terms{I}, t::Term{I}) where {I} = a + (-t)

Base.one(::Terms{I}) where {I} = Terms{I}(Term{I}[Term{I}(Int[], ITOKey{I}[], ComplexF64(1))])

"""
    isapprox(a::Terms, b::Terms; kwargs...)
    a ≈ b

Whether two operators carry the same terms with matching coefficients: the canonical term sets must
be **equal** (a dropped term is never "approximately" absent) and the coefficients `≈`. No lattice is
involved, so an operator reconstructed by [`mpo_terms`](@ref) — which knows the bonds but not the
physical spaces — compares equal to the one it came from. This is the faithfulness check.
"""
Base.isapprox(a::Terms{I}, b::Terms{I}; kwargs...) where {I} =
    _termsapprox(canonicalize!(a).terms, canonicalize!(b).terms; kwargs...)
Base.:(==)(a::Terms{I}, b::Terms{I}) where {I} =
    _termsequal(canonicalize!(a).terms, canonicalize!(b).terms)

function Base.show(io::IO, ts::Terms)
    canonicalize!(ts)
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

_tolattice(lat::AbstractLattice) = lat
_tolattice(sites::AbstractVector{<:ElementarySpace}) = FiniteChain(sites)
function _tolattice(sites)
    # Guard on iterability first: without it a misplaced argument (an algorithm selector, say) is
    # `collect`ed and fails deep inside Base instead of naming what was wrong.
    applicable(iterate, sites) || _notalattice(sites)
    lat = collect(sites)
    eltype(lat) <: ElementarySpace || _notalattice(sites)
    return FiniteChain(lat)
end

_notalattice(sites) = throw(
    ArgumentError(
        "a lattice must be a `FiniteChain`, an `InfiniteChain`, or an iterable of " *
            "`ElementarySpace`s (one per site), got $(typeof(sites))"
    )
)

# The one place operators and the spaces they are compressed with are confronted; without it a
# lattice from a different model silently gives a wrong MPO. Every entry point that takes a lattice
# runs it, which is where it moved to when `opsum` stopped taking one — still exactly one place per
# call, and still before any output is produced.
#
# `Θ(Σ arity)`, not `Θ(N)`. On an `InfiniteChain` there is no site range to check (a generating term
# legitimately reaches past the cell) and `lat[s]` wraps.
function _checklattice(terms::AbstractVector{Term{I}}, lat::AbstractLattice) where {I}
    isempty(terms) && return nothing
    # An operator over a different symmetry than the lattice would otherwise fail deep inside the
    # alphabet lookup, on a charge of the wrong type.
    J = _lattice_sectortype(lat)
    J === I || throw(
        ArgumentError("cannot compress an operator over $I on a lattice of sector type $J")
    )
    N = _maxsite(lat)
    seen = Dict{eltype(lat), Dict{I, Int}}()
    for t in terms
        for (s, key) in zip(t.sites, t.keys)
            (N === nothing || 1 <= s <= N) || throw(
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

_checklattice(ts::Terms, lat::AbstractLattice) = _checklattice(ts.terms, lat)

"""
    opsum(terms...) -> Terms

Accumulate terms into one operator, in **one pass** over each argument, so building an `M`-term
Hamiltonian costs `Θ(M)`:

```julia
h = opsum(J * dot(S[i], S[i + 1]) for i in 1:(N - 1))
```

Each argument may be a [`Term`](@ref), a [`Terms`](@ref) bag, or any iterable of those, nested
arbitrarily. The result is latticeless: the lattice is supplied where the MPO is formed
([`irrep_mpo`](@ref)), which is also where every letter is checked against the space of the site it
acts on.

`opsum` is the linear route; `+` *copies*, so folding it over `M` terms is quadratic.
[`append!`](@ref) adds to an existing bag, also in one pass.
"""
function opsum(args...)
    isempty(args) && throw(
        ArgumentError(
            "opsum() has no terms, so the symmetry sector cannot be inferred; write `Terms{I}()` " *
                "for the empty operator"
        )
    )
    out = nothing
    for a in args
        out = _collect_any(out, a)
    end
    out === nothing && throw(
        ArgumentError(
            "opsum: every argument was empty, so the symmetry sector cannot be inferred; write " *
                "`Terms{I}()` for the empty operator"
        )
    )
    return Terms(out)
end

# `opsum` no longer binds a lattice. A bare vector of spaces as the first argument is the old
# signature, so name the replacement instead of failing on the element type.
opsum(::AbstractLattice, args...) = _nolattice()
opsum(::AbstractVector{<:ElementarySpace}, args...) = _nolattice()
_nolattice() = throw(
    ArgumentError(
        "opsum no longer takes a lattice: a term bag is latticeless, and the lattice is supplied " *
            "where the MPO is formed. Write `opsum(terms...)` and pass the lattice to " *
            "`irrep_mpo(h, lat)` / `instantiate(h, lat)` / `islossless(h, lat)`."
    )
)

# Establishing `I`: the accumulator is `nothing` until the first term fixes the sector type. Only
# that one step is dynamically dispatched; everything after it runs through the typed collector, so
# `opsum` stays `Θ(M)` with the same constants as when the lattice supplied `I`.
_collect_any(::Nothing, t::Term{I}) where {I} = Term{I}[t]
_collect_any(::Nothing, ts::Terms{I}) where {I} = copy(ts.terms)
function _collect_any(::Nothing, itr)
    out = nothing
    for a in itr
        out = _collect_any(out, a)
    end
    return out
end
_collect_any(out::Vector{Term{I}}, a) where {I} = (_collect_terms!(out, a, I); out)

_collect_terms!(out::Vector{Term{I}}, t::Term{I}, ::Type{I}) where {I} = push!(out, t)
_collect_terms!(out::Vector{Term{I}}, ts::Terms{I}, ::Type{I}) where {I} =
    append!(out, ts.terms)
function _collect_terms!(out::Vector{Term{I}}, itr, ::Type{I}) where {I}
    for a in itr
        _collect_terms!(out, a, I)
    end
    return out
end
# A term over a different symmetry than the one already accumulated is a mistake worth naming.
_collect_terms!(::Vector{Term{I}}, ::Term{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_collect_terms!(::Vector{Term{I}}, ::Terms{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_wrongsector(I, J) = throw(
    ArgumentError("cannot combine an operator over $J with one over $I")
)

"""
    append!(ts::Terms, terms...) -> ts

Append terms to `ts` **in place**, in one pass — the linear way to accumulate into an existing
operator. Same argument forms as [`opsum`](@ref).
"""
function Base.append!(ts::Terms{I}, args...) where {I}
    for a in args
        _collect_terms!(ts.terms, a, I)
    end
    return ts
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

# Whether `b`'s leg may be *inserted* to the left of some of `a`'s, instead of only appended.
#
# `UniqueFusion` is what makes it well defined without F-moves, twice over: every running bond charge
# the insertion invalidates has a single forced replacement, and every leg transposition is a single
# scalar `Rsymbol(x, y, x ⊗ y)`. `SymmetricBraiding` (`R = R⁻¹ = ±1`) is what makes the answer
# independent of whether the leg is braided over or under — a convention this API does not expose.
_canreorder(::Type{I}) where {I <: Sector} =
    FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding

# Extend every term of `a` by every single-site term of `b`, fusing to `target(running total, b's
# letter)`; unreachable pairs are dropped, so the result may be empty.
#
# Storage is site-ordered. When `b` acts to the right of all of `a` this is a plain append: extending
# a caterpillar leaves the earlier running charges alone, so only the channel's existence has to be
# checked (one `Nsymbol`) and no tree is built. When `b` acts further left its leg is *inserted* at
# position `p` instead (see `_canreorder`): the running bond charges from `p` on are recomputed
# (forced, by unique fusion) and the coefficient picks up one R-symbol per leg of `a` that `b` braids
# past — for fermions, exactly the anticommutation sign.
function _couple_terms(a::Terms{I}, b::Terms{I}, target) where {I}
    out = Term{I}[]
    sizehint!(out, length(a) * length(b))
    reorder = _canreorder(I)
    for tb in b.terms
        arity(tb) == 1 || throw(
            ArgumentError("couple: every term of the second operand must be a single-site charged operator")
        )
    end
    for ta in a.terms
        na = arity(ta)
        na >= 1 || throw(
            ArgumentError("couple: every term of the first operand must carry at least one charged operator")
        )
        run = last(ta.keys).bond
        for tb in b.terms
            sb = only(tb.sites)
            opb = only(tb.keys).op

            # where `b`'s site falls among `a`'s: the first of them strictly to its right
            p = na + 1
            for j in 1:na
                s = ta.sites[j]
                s == sb && throw(ArgumentError("couple: operators must act on distinct sites"))
                s > sb && (p = j; break)
            end
            (p > na || reorder) || throw(
                ArgumentError(
                    "couple: the second operand acts on site $sb, left of site $(ta.sites[p]) of " *
                        "the first, and reordering the legs of a $(FusionStyle(I)) coupling needs " *
                        "F-moves, which is deferred. Write the operands in increasing site order."
                )
            )

            tot = target(run, opb)::I
            nsym = Nsymbol(run, opb.c, tot)
            iszero(nsym) && continue    # this pair of charges cannot fuse to `tot`: drop it
            @assert isone(nsym) "expected a unique coupling channel; multi-channel (GenericFusion) coupling is deferred"

            sites = Vector{Int}(undef, na + 1)
            keys = Vector{ITOKey{I}}(undef, na + 1)
            coeff = ta.coeff * tb.coeff
            for j in 1:(p - 1)
                sites[j] = ta.sites[j]
                keys[j] = ta.keys[j]
            end
            sites[p] = sb
            if p == na + 1
                keys[p] = ITOKey{I}(opb, tot, 1)
            else
                bond = p == 1 ? opb.c : only(ta.keys[p - 1].bond ⊗ opb.c)
                keys[p] = ITOKey{I}(opb, bond, 1)
                for j in p:na
                    op = ta.keys[j].op
                    coeff *= Rsymbol(op.c, opb.c, only(op.c ⊗ opb.c))
                    bond = only(bond ⊗ op.c)
                    sites[j + 1] = ta.sites[j]
                    keys[j + 1] = ITOKey{I}(op, bond, 1)
                end
                # unique fusion is commutative, so the reordered chain lands on the same total as the
                # appended one would have; a mismatch means the recomputation drifted
                bond == tot ||
                    _invariant("reordered coupling reached $bond, not the total $tot")
            end
            push!(out, Term{I}(sites, keys, coeff))
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

Under an **abelian** symmetry the operands may be written in any site order: the stored form is
site-ordered, so an out-of-order leg is inserted rather than appended, and the braiding phase that
costs is inserted with it. For a fermionic sector that phase *is* the anticommutation sign, so

```julia
couple(cd[i + 1], c[i]) == -couple(c[i], cd[i + 1])       # both spellings are available
```

so a hand-written sign is now only a way to get it wrong. Under a
non-abelian symmetry reordering needs F-moves and still throws: each operand after the first must
then act to the **right** of every site before it. ([`dot`](@ref) accepts either order for any
symmetry — with only two legs coupling to the unit sector, no F-move arises.)

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
    for i in 1:(length(others) - 1)
        nxt = others[i]
        isempty(nxt) &&
            throw(ArgumentError("couple: operand $(i + 1) has no terms"))
        # intermediates are forced by unique fusion; only the last step has to land on `to`, which is
        # the 2-arg method's job
        acc = _couple_terms(acc, nxt, (ta, opb) -> only(ta ⊗ opb.c))
        isempty(acc) && throw(
            ArgumentError("couple: no pair of terms of operands 1..$(i + 1) can be coupled")
        )
    end
    lastop = last(others)
    isempty(lastop) &&
        throw(ArgumentError("couple: operand $(length(others) + 1) has no terms"))
    return couple(acc, lastop; to = tot)
end

"""
    dot(a::Terms, b::Terms)
    a · b

The Cartesian two-body scalar product of two single-site ITO operators: singlet coupling
`couple(a, b; to = unit(I))` times the Cartesian factor `-√dim(c)` (the identity
`Sᵢ·Sⱼ = -√3 [S⊗S]⁽⁰⁾` with `-√3 = -√dim(spin-1)`).

Either site order is accepted, for **any** symmetry: two legs coupling to the unit sector is the one
case where reordering needs no F-move, whatever the fusion style, so `dot` is defined where the
general out-of-order [`couple`](@ref) is not. The Cartesian factor is read off the operator that ends
up on the left, which changes nothing — the two charges must fuse to the unit sector, i.e. be each
other's dual, and dual sectors have equal quantum dimension. The braiding phase does *not* always
cancel, though: `dot(a, b) == R · dot(b, a)` with `R = Rsymbol(cₐ, c_b, unit)`, which is `+1` for
bosonic integer charges (so `spin`, and hence every spin chain here, is order-independent) and `-1`
for a pair of odd fermionic charges or of half-integer SU(2) charges.

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
    sa < sb && return scale(couple(a, b; to = unit(I)), -sqrt(dim(opa.c)))

    # `b` sits to the left, so the two legs have to be swapped into storage order. With two legs
    # coupling to the unit sector that swap is one scalar R-symbol for any multiplicity-free fusion
    # style — no F-move, which is why this works under SU(2) where the general `couple` reordering
    # does not. Restricted to symmetric braiding so that `R = R⁻¹` and the (unexposed) over/under
    # convention cannot change the answer.
    BraidingStyle(I) isa SymmetricBraiding || throw(
        ArgumentError(
            "·: the operands are in decreasing site order and $I has $(BraidingStyle(I)) braiding, " *
                "where the swap phase depends on a braiding direction this API does not expose; " *
                "write the operands in increasing site order"
        )
    )
    R = Rsymbol(opa.c, opb.c, unit(I))
    return scale(couple(b, a; to = unit(I)), -sqrt(dim(opb.c)) * R)
end
