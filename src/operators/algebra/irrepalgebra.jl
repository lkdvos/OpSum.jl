# Term algebra over ITOs: `Term` (one term) → `Terms` (a bag, the operator). Latticeless: the
# physical spaces enter with the `OperatorSum`. `GenericFusion` multiplicity is deferred.

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

Materialize a single-site operator as the coefficient-weighted sum of its letters' tensors,
`V ← V ⊗ Vect[I](c => 1)`. The pass-through letter is the identity with a trivial charge leg.
"""
function instantiate(O::SiteOperator{I}, V::ElementarySpace) where {I}
    isempty(O) && throw(ArgumentError("cannot instantiate an empty SiteOperator without a space"))
    return sum(((l, c),) -> c * instantiate(l, V), pairs(O))
end

"""
    scalarop(c::Number, B::Basis)
    scalarop(c::Number, ::Type{I<:Sector})

A scalar local operator `c·𝟙`: `c` times the identity letter `(unit(I), 1)`, element 1 of every basis.
"""
scalarop(c::Number, ::Type{I}) where {I} = SiteOperator{I}(ComplexF64(c))
scalarop(c::Number, V::ElementarySpace) = scalarop(c, sectortype(V))

# The 4-arg `FusionTree` form: the 3-arg one throws for `MultipleFusion`.
_idtree(::Type{I}) where {I <: Sector} = FusionTree{I}((), unit(I), (), ())

# Inverse of `bondcharges` (multiplicity-free: all vertex labels 1).
function _tree_from_bonds(charges::AbstractVector{I}, bonds::AbstractVector{I}) where {I <: Sector}
    K = length(charges)
    K == 0 && return _idtree(I)
    unc = ntuple(i -> charges[i], K)
    isd = ntuple(_ -> false, K)
    inner = ntuple(i -> bonds[i + 1], max(0, K - 2))   # innerlines = bonds[2 : K-1]
    vtx = ntuple(_ -> 1, max(0, K - 1))
    return FusionTree{I}(unc, bonds[K], isd, inner, vtx)
end

"""
    Term{I<:Sector}

A single term: `sites` (ascending, unique), one [`ITOKey`](@ref) each (letter, running bond charge and
vertex label), and a reduced `coeff`. The caterpillar coupling tree is derived from the running bond
charges ([`tree`](@ref)). Multiplicity-free fusion only.

Never mutated, so terms may share vectors. `==`/`hash` ignore the coefficient, which is what makes
summing coincident terms well defined.
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

The total charge of `t`: the running bond charge after its last site (`unit` for the identity term).
"""
total(t::Term{I}) where {I} = isempty(t.keys) ? unit(I) : last(t.keys).bond

"""
    tree(t::Term) -> FusionTree

The left-nested (caterpillar) coupling tree of `t`, rebuilt from its running bond charges and vertex
labels (the same information, irrepkey.jl).
"""
function tree(t::Term{I}) where {I}
    arity(t) == 0 && return _idtree(I)
    return _tree_from_bonds(I[k.op.c for k in t.keys], I[k.bond for k in t.keys])
end

Base.:(==)(x::Term{I}, y::Term{I}) where {I} = x.sites == y.sites && x.keys == y.keys
Base.hash(t::Term, h::UInt) = hash(t.keys, hash(t.sites, hash(:ITOTerm, h)))

# Total order for the normal form; the keys distinguish letters *and* couplings.
function Base.isless(x::Term{I}, y::Term{I}) where {I}
    x.sites == y.sites || return isless(x.sites, y.sites)
    return isless(x.keys, y.keys)
end

VectorInterface.scale(t::Term{I}, α::Number) where {I} =
    Term{I}(t.sites, t.keys, t.coeff * ComplexF64(α))

function Base.show(io::IO, t::Term)
    return print(io, "Term(sites=", t.sites, ", ops=", ops(t), ", total=", total(t), ", coeff=", t.coeff, ")")
end

"""
    PendingTerm{I<:Sector}

A product acting more than once on one site (`a[i] * b[i]`, `hop[i, i]`), kept unevaluated until it
enters an [`OperatorSum`](@ref), where it is resolved in the sites' bases (products.jl). The letters
are in written order with the running bond charges of the written-order caterpillar; the operator is
`coeff` times that caterpillar's product.
"""
struct PendingTerm{I <: Sector}
    sites::Vector{Int}
    keys::Vector{ITOKey{I}}
    coeff::ComplexF64
end

"""
    Terms{I<:Sector}

The compressible ITO operator: a bag of [`Term`](@ref)s with **no lattice**. What `A[i]` and `B[i]`
(placing a [`SiteOperator`](@ref) or a [`LocalOperator`](@ref)), [`couple`](@ref) and `dot` return,
what `+`, `-`, `*` and `/` combine, and what [`opsum`](@ref) accumulates in one pass. The lattice
enters when the terms are added to an [`OperatorSum`](@ref).

Terms are appended as given; [`canonicalize!`](@ref) normalises in place. `length`, iteration,
indexing and `≈` go through it, so they report the *canonical* operator (`nterms_raw(ts)` is the
append count, for tests).
"""
struct Terms{I <: Sector}
    terms::Vector{Term{I}}
    pending::Vector{PendingTerm{I}}
end
Terms{I}(terms::Vector{Term{I}}) where {I <: Sector} = Terms{I}(terms, PendingTerm{I}[])
Terms{I}() where {I <: Sector} = Terms{I}(Term{I}[])
Terms(terms::Vector{Term{I}}) where {I} = Terms{I}(terms)
Terms(t::Term{I}) where {I} = Terms{I}(Term{I}[t])

sectortype(::Type{Terms{I}}) where {I} = I

"""
    nterms_raw(ts::Terms) -> Int

The number of *appended* terms, before coincident ones are summed. For tests.
"""
nterms_raw(ts::Terms) = length(ts.terms)

# Everything that observes the term set normalises first.
Base.length(ts::Terms) = length(canonicalize!(ts).terms)
Base.isempty(ts::Terms) = isempty(canonicalize!(ts).terms)
Base.iterate(ts::Terms, args...) = iterate(canonicalize!(ts).terms, args...)
Base.eltype(::Type{Terms{I}}) where {I} = Term{I}
Base.getindex(ts::Terms, i::Integer) = canonicalize!(ts).terms[i]
Base.firstindex(::Terms) = 1
Base.lastindex(ts::Terms) = length(ts)

# Copies, so folding over M terms is quadratic; use `opsum`.
Base.:+(a::Terms{I}, b::Terms{I}) where {I} =
    Terms{I}(vcat(a.terms, b.terms), vcat(a.pending, b.pending))
Base.:+(a::Terms{I}, t::Term{I}) where {I} = Terms{I}(vcat(a.terms, [t]), copy(a.pending))
Base.:+(t::Term{I}, a::Terms{I}) where {I} = Terms{I}(vcat([t], a.terms), copy(a.pending))
VectorInterface.scale(ts::Terms{I}, α::Number) where {I} = Terms{I}(
    Term{I}[scale(t, α) for t in ts.terms],
    PendingTerm{I}[PendingTerm{I}(p.sites, p.keys, p.coeff * ComplexF64(α)) for p in ts.pending]
)

"""
    one(::Terms{I})
    one(::Type{Terms{I}})

The identity operator: the single term with no active sites. `zero` is the empty bag.
"""
Base.one(::Type{Terms{I}}) where {I} =
    Terms{I}(Term{I}[Term{I}(Int[], ITOKey{I}[], ComplexF64(1))])
Base.one(::Terms{I}) where {I} = one(Terms{I})

"""
    zero(::Terms{I})
    zero(::Type{Terms{I}})
    empty(::Terms{I})

A fresh operator with no terms, ready to [`append!`](@ref) into.
"""
Base.zero(::Type{Terms{I}}) where {I} = Terms{I}()
Base.zero(::Terms{I}) where {I} = Terms{I}()
Base.empty(::Terms{I}) where {I} = Terms{I}()

"""
    copy(ts::Terms) -> Terms

A shallow copy with its own term vectors; the (immutable) [`Term`](@ref)s are shared.
"""
Base.copy(ts::Terms{I}) where {I} = Terms{I}(copy(ts.terms), copy(ts.pending))

"""
    isapprox(a::Terms, b::Terms; kwargs...)
    a ≈ b

Whether two operators are close as sparse vectors over their canonical terms (a missing term counts
as zero): `‖a - b‖ ≤ max(atol, rtol · max(‖a‖, ‖b‖))`; `==` compares exactly. No lattice is involved,
so an operator reconstructed by [`mpo_terms`](@ref) compares equal to the original: the faithfulness
check.
"""
Base.isapprox(a::Terms{I}, b::Terms{I}; kwargs...) where {I} =
    _termsapprox(canonicalize!(a).terms, canonicalize!(b).terms; kwargs...)
Base.:(==)(a::Terms{I}, b::Terms{I}) where {I} =
    _termsequal(canonicalize!(a).terms, canonicalize!(b).terms)

function Base.show(io::IO, ts::Terms{I}) where {I}
    isempty(ts.pending) || return print(
        io, "Terms{", I, "}(", length(ts.terms), " term(s) and ", length(ts.pending),
        " unresolved on-site product(s); add it to an OperatorSum to resolve them)"
    )
    canonicalize!(ts)
    print(io, "Terms(")
    join(io, ("$(t.coeff) * $(_termbody(t))" for t in ts.terms), " + ")
    return print(io, ")")
end
_termbody(t::Term) = string("[sites=", t.sites, ", ops=", ops(t), ", total=", total(t), "]")

_tolattice(lat::AbstractLattice) = lat
function _tolattice(sites)
    applicable(iterate, sites) ||
        throw(ArgumentError("a lattice must be a `FiniteChain`, an `InfiniteChain`, or an iterable of operator `Basis`es, got $(typeof(sites))"))
    return FiniteChain(vec(collect(sites)))
end

# Confronts operators with the lattice's bases when terms enter an `OperatorSum`, so the error points
# at the line that added the term. `Θ(Σ arity)`; on an `InfiniteChain` there is no site range to check.
function _checklattice(terms::AbstractVector{Term{I}}, lat::AbstractLattice) where {I}
    isempty(terms) && return nothing
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
                    "letter $op does not exist on site $s ($V): that site carries " *
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

Accumulate terms into one operator in **one pass** over each argument (`Θ(M)` for `M` terms):

```julia
SS = spin_ops(SU2Irrep).S_exchange
h = opsum(J * SS[i, i + 1] for i in 1:(N - 1))
```

Each argument may be a [`Term`](@ref), a [`Terms`](@ref) bag, or any iterable of those, nested
arbitrarily. The result is latticeless; `opsum(lat, terms...)` accumulates into an
[`OperatorSum`](@ref), which checks every letter against its site's basis. `+` copies, so folding it
is quadratic; [`append!`](@ref) adds to an existing bag in one pass.
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
    return out
end

# The accumulator is `nothing` until the first term fixes `I`; only that step is dynamically
# dispatched, the rest runs through the typed collector.
_collect_any(::Nothing, t::Term{I}) where {I} = Terms{I}(Term{I}[t])
_collect_any(::Nothing, ts::Terms{I}) where {I} = copy(ts)
function _collect_any(::Nothing, itr)
    out = nothing
    for a in itr
        out = _collect_any(out, a)
    end
    return out
end
_collect_any(out::Terms{I}, a) where {I} = (_collect_terms!(out, a, I); out)
# a bare space is iterable, so a vector of them would otherwise be read as a nest of terms
_collect_any(::Nothing, ::ElementarySpace) = _spaceaslattice()
_collect_terms!(out::Terms{I}, ::ElementarySpace, ::Type{I}) where {I} = _spaceaslattice()
_spaceaslattice() = throw(
    ArgumentError(
        "opsum: got an ElementarySpace where an operator was expected; a lattice is one operator " *
            "`Basis` per site (e.g. `FiniteChain(spin_basis(SU2Irrep), N)`), not a list of spaces"
    )
)

_collect_terms!(out::Terms{I}, t::Term{I}, ::Type{I}) where {I} = (push!(out.terms, t); out)
function _collect_terms!(out::Terms{I}, ts::Terms{I}, ::Type{I}) where {I}
    append!(out.terms, ts.terms)
    append!(out.pending, ts.pending)
    return out
end
function _collect_terms!(out::Terms{I}, itr, ::Type{I}) where {I}
    for a in itr
        _collect_terms!(out, a, I)
    end
    return out
end
# A term over a different symmetry than the one already accumulated is a mistake worth naming.
_collect_terms!(::Terms{I}, ::Term{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_collect_terms!(::Terms{I}, ::Terms{J}, ::Type{I}) where {I, J} = _wrongsector(I, J)
_wrongsector(I, J) = throw(
    ArgumentError("cannot combine an operator over $J with one over $I")
)

"""
    append!(ts::Terms, terms...) -> ts

Append terms to `ts` **in place**, in one pass. Same argument forms as [`opsum`](@ref).
"""
function Base.append!(ts::Terms{I}, args...) where {I}
    for a in args
        _collect_terms!(ts, a, I)
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
        # pass-through is the bare identity: a K = 0 term
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

# Operand of `dot`: a single-site charged term; returns `(site, op)`.
function _single_site_term(b::Terms, ctx)
    length(b) == 1 || throw(ArgumentError("$ctx must be a single-term operator"))
    t = only(b.terms)
    arity(t) == 1 || throw(ArgumentError("$ctx must be a single-site charged operator"))
    return only(t.sites), only(t.keys).op
end

# Whether `b`'s leg may be *inserted* left of some of `a`'s. `UniqueFusion` forces every invalidated
# running bond charge and makes each leg transposition one scalar `Rsymbol(x, y, x ⊗ y)`;
# `SymmetricBraiding` (`R = R⁻¹`) makes the answer independent of braiding over or under.
_canreorder(::Type{I}) where {I <: Sector} =
    FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding

# Fermion-odd: a fermionic sector whose twist is -1 (not SU(2)'s half-integer spin, which is bosonic).
_isfermionodd(c::Sector) = BraidingStyle(typeof(c)) isa Fermionic && real(twist(c)) < 0

# Extend every term of `a` by every single-site term of `b`, fusing to `target(running total, b's
# letter)`; unreachable pairs are dropped, so the result may be empty.
#
# Storage is site-ordered. If `b` is right of all of `a` this is a plain append (one `Nsymbol` check).
# Otherwise its leg is *inserted* at position `p` (see `_canreorder`): the running bond charges from
# `p` on are recomputed and the coefficient picks up one R-symbol per leg `b` braids past (for
# fermions, the anticommutation sign). Under a non-abelian symmetry the leg is appended in written
# order and re-coupled into site order (`_recouple`), which may give several terms.
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

            tgt = target(run, opb)
            tgt === nothing && continue    # no channel of this pair reaches the requested total
            tot = tgt::I
            nsym = Nsymbol(run, opb.c, tot)
            iszero(nsym) && continue
            @assert isone(nsym) "expected a unique coupling channel; multi-channel (GenericFusion) coupling is deferred"

            coeff = ta.coeff * tb.coeff
            # graded tensor product: -1 iff both are fermion-odd
            _isfermionodd(run) && _isfermionodd(opb.c) && (coeff = -coeff)
            if p <= na && !reorder
                # non-abelian: append in written order, then re-couple into site order
                keys_w = vcat(ta.keys, ITOKey{I}(opb, tot, 1))
                append!(out, _recouple(vcat(ta.sites, sb), keys_w, coeff))
                continue
            end
            sites = Vector{Int}(undef, na + 1)
            keys = Vector{ITOKey{I}}(undef, na + 1)
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
    couple(a::Terms, b::Terms, cs::Terms...; to = unit(I))   # channels must be forced

Left-nested (caterpillar) irrep coupling: extend the composite `a` (K ≥ 1 sites, with its running
coupling charges) by one single-site operator `b`, fusing the running total of `a` with `b`'s charge
to `to` at a new vertex. `to` defaults to the unit sector (a Hamiltonian term); pass it explicitly to
build a charged object. It is the primitive under `*` and placement. Its operands are *charged*
single-site operators: a basis' charged elements (`project(t, B)` of a charged `t : V ← V ⊗ Vect[I](c => 1)`).
Below, `cd` and `c` are a fermion's creation and annihilation operators and `S` the SU(2) spin vector.

The operands may be written in any site order: an out-of-order leg is inserted rather than appended,
with the braiding phase that costs (for fermions the anticommutation sign), so
`couple(cd[i + 1], c[i]) == -couple(c[i], cd[i + 1])`. Under a non-abelian symmetry the coupling is
formed in written order and re-coupled into site order, so one pair of terms can give several, with
F- and R-symbol coefficients.

Both operands may be composite (several terms, e.g. from [`project`](@ref)): the coupling distributes
over every pair, and pairs whose charges cannot fuse to `to` are dropped. It is an error if *no* pair
fuses. Each term of `a` must carry at least one charged operator, each of `b` be one charged site.

The variadic form takes the intermediate channels the charges force, and errors only where a channel
is a genuine choice. Under an abelian symmetry (`UniqueFusion`) every one is forced; under a
non-abelian one some still are:

```julia
couple(cd[1], c[2], cd[3], c[4])        # a charge-neutral four-fermion term
couple(S[1], S[2], S[3])                # j₁₂ = 1 is the only singlet channel
```

Where the channels are free the error lists them ([`couple_channels`](@ref) returns them as a value)
and you nest, naming each one:

```julia
couple_channels(S[1], S[2], S[3], S[4])                                  # [(0,1), (1,1), (2,1)]
couple(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(1)), S[4])
```

For fermions `couple(cd[i], c[j])` is the physical `c†ᵢcⱼ` (Jordan–Wigner, sites `1..N`). This is the
bare fusion coupler: reduced coeff `= va·vb`, no normalization (the Cartesian convention lives in
[`dot`](@ref)). `GenericFusion` and tree-structured (`via`) coupling are deferred.
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

# The distinct running totals of the accumulator's terms, and the distinct letter charges of each
# later operand: all the channel arithmetic needs.
function _startcharges(a::Terms{I}) where {I}
    isempty(a) && throw(ArgumentError("couple: first operand has no terms"))
    cs = I[]
    for t in a
        arity(t) >= 1 || throw(
            ArgumentError("couple: every term of the first operand must carry at least one charged operator")
        )
        c = total(t)
        c in cs || push!(cs, c)
    end
    return cs
end

function _legcharges(b::Terms{I}, which::Int) where {I}
    isempty(b) && throw(ArgumentError("couple: operand $which has no terms"))
    cs = I[]
    for t in b
        arity(t) == 1 || throw(
            ArgumentError(
                "couple: operand $which must be a single-site charged operator in every term"
            )
        )
        c = only(t.keys).op.c
        c in cs || push!(cs, c)
    end
    return cs
end

# One charge from each of `legs[i:end]`, every way round.
_tails(legs::Vector{Vector{I}}, i::Int) where {I <: Sector} =
    Iterators.product(view(legs, i:lastindex(legs))...)

# Whether `d` can still reach `tot` by fusing one charge from each of `legs[i:end]`. Memoised on
# `(charge, position)`; past the last leg the product is one empty tuple, which is the base case.
function _reaches(
        d::I, legs::Vector{Vector{I}}, i::Int, tot::I, memo::Dict{Tuple{I, Int}, Bool}
    ) where {I <: Sector}
    return get!(memo, (d, i)) do
        any(cs -> !isempty(caterpillar_trees((d, cs...), tot)), _tails(legs, i))
    end
end

# Every legal tuple of intermediate channels (a caterpillar tree's `innerlines`), one per vertex but the last.
function _channeltuples(starts::Vector{I}, legs::Vector{Vector{I}}, tot::I) where {I <: Sector}
    out = NTuple{length(legs) - 1, I}[]
    for (s, cs) in Iterators.product(starts, _tails(legs, 1)), f in caterpillar_trees((s, cs...), tot)
        f.innerlines in out || push!(out, f.innerlines)
    end
    return out
end

"""
    couple_channels(a::Terms, ops::Terms...; to = unit(I)) -> Vector{NTuple{N, I}}

The intermediate coupling channels a caterpillar [`couple`](@ref) of these operands may legally use,
one tuple of `N = length(ops) - 1` charges per channel (the `to` of each nested `couple` but the
outermost). `N == 0` gives `[()]` when the two operands fuse at all and `[]` when they do not.

The variadic `couple` needs no channel argument exactly when this returns a single tuple:

```julia
couple_channels(S[1], S[2], S[3]; to = SU2Irrep(0))      # [(1,)] — forced
couple_channels(S[1], S[2], S[3], S[4])                  # [(0,1), (1,1), (2,1)] — three singlets
```

Composite operands contribute every charge they carry.
"""
function couple_channels(a::Terms{I}, rest::Terms{I}...; to = unit(I)) where {I}
    isempty(rest) &&
        throw(ArgumentError("couple_channels: needs at least two operands"))
    tot = to::I
    starts = _startcharges(a)
    legs = Vector{I}[_legcharges(o, i + 1) for (i, o) in enumerate(rest)]
    return _channeltuples(starts, legs, tot)
end

# The vertex charge at one fold step: of `run ⊗ c`, the products that can still reach `tot`.
# `nothing` drops the pair; more than one is a real choice (error).
function _forcedchannel(
        run::I, c::I, legs::Vector{Vector{I}}, i::Int, tot::I, starts::Vector{I},
        memo::Dict{Tuple{I, Int}, Bool}
    ) where {I <: Sector}
    hit = nothing
    for d in run ⊗ c
        _reaches(d, legs, i, tot, memo) || continue
        hit === nothing || _ambiguouschannels(starts, legs, tot)
        hit = d
    end
    return hit
end

# Variadic coupling: fold left, taking each intermediate channel the charges force.
function couple(
        a::Terms{I}, b::Terms{I}, c::Terms{I}, rest::Terms{I}...;
        to = unit(I), via = nothing
    ) where {I}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    FusionStyle(I) isa GenericFusion && throw(
        ArgumentError(
            "couple: multi-channel (GenericFusion) coupling is deferred, and $I has " *
                "$(FusionStyle(I))"
        )
    )
    tot = to::I

    others = (b, c, rest...)
    starts = _startcharges(a)
    legs = Vector{I}[_legcharges(o, i + 1) for (i, o) in enumerate(others)]
    memo = Dict{Tuple{I, Int}, Bool}()

    acc = a
    for i in 1:(length(others) - 1)
        acc = _couple_terms(
            acc, others[i],
            (run, opb) -> _forcedchannel(run, opb.c, legs, i + 1, tot, starts, memo)
        )
        isempty(acc) && _nochannel(starts, legs, tot, i + 1)
    end
    return couple(acc, last(others); to = tot)
end

@noinline function _nochannel(
        starts::Vector{I}, legs::Vector{Vector{I}}, tot::I, upto::Int
    ) where {I}
    return throw(
        ArgumentError(
            "couple: no terms of operands 1..$upto can be coupled towards $tot — the charges " *
                "admit $(length(_channeltuples(starts, legs, tot))) channel tuple(s) to $tot " *
                "(`couple_channels` lists them)"
        )
    )
end

# Only reached on a genuine choice, so the enumeration is paid for in the error only.
@noinline function _ambiguouschannels(starts::Vector{I}, legs::Vector{Vector{I}}, tot::I) where {I}
    channels = _channeltuples(starts, legs, tot)
    example = isempty(channels) ? "j₁₂" : string(first(first(channels)))
    return throw(
        ArgumentError(
            "couple: the intermediate channels are a genuine choice here, so the variadic form " *
                "cannot pick one — $(length(channels)) tuple(s) couple these operands to $tot, " *
                "namely $(channels). Nest `couple` to name each channel, e.g. " *
                "couple(couple(a, b; to = $(example)), c; to = $tot); " *
                "`couple_channels` returns the legal tuples."
        )
    )
end

"""
    dot(a::Terms, b::Terms)
    a · b

The Cartesian two-body scalar product of two single-site ITO operators: singlet coupling
`couple(a, b; to = unit(I))` times the Cartesian factor `-√dim(c)` (`Sᵢ·Sⱼ = -√3 [S⊗S]⁽⁰⁾`).

Either site order is accepted, for **any** symmetry: two legs coupling to the unit sector need no
F-move to reorder. `dot(a, b) == R · dot(b, a)` with `R = Rsymbol(cₐ, c_b, unit)`, which is `+1` for
bosonic integer charges (so spin chains are order-independent) and `-1` for a pair of odd fermionic
or half-integer SU(2) charges.

Unlike [`couple`](@ref) this does not distribute over composite operands: the factor `-√dim(c)` is
per-letter.
"""
function LinearAlgebra.dot(a::Terms{I}, b::Terms{I}) where {I}
    sa, opa = _single_site_term(a, "·: first operand")
    sb, opb = _single_site_term(b, "·: second operand")
    sa == sb && throw(ArgumentError("·: operators must act on distinct sites"))
    iszero(Nsymbol(opa.c, opb.c, unit(I))) && throw(
        ArgumentError(
            "·: the two operator charges $(opa.c) and $(opb.c) do not fuse to the unit sector, so " *
                "there is no scalar product; use `couple(a, b; to = …)` for a charged coupling"
        )
    )
    sa < sb && return scale(couple(a, b; to = unit(I)), -sqrt(dim(opa.c)))

    # `b` is to the left: swap the legs into storage order with one scalar R-symbol (no F-move,
    # which is why this works under SU(2)). Symmetric braiding keeps `R = R⁻¹`, so the over/under
    # convention cannot matter.
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
