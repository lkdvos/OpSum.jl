# Term algebra over ITOs: `Term` (one term) → `Terms` (a bag, the operator). Latticeless.

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

The coefficient-weighted sum of `O`'s letters' tensors, `V ← V ⊗ Vect[I](c => 1)`.
"""
function instantiate(O::SiteOperator{I}, V::ElementarySpace) where {I}
    isempty(O) && throw(ArgumentError("cannot instantiate an empty SiteOperator without a space"))
    return sum(((l, c),) -> c * instantiate(l, V), pairs(O))
end

"""
    scalarop(c::Number, B::Basis)
    scalarop(c::Number, ::Type{I<:Sector})

The scalar operator `c·𝟙`: `c` times the identity letter `(unit(I), 1)`, element 1 of every basis.
"""
scalarop(c::Number, ::Type{I}) where {I} = SiteOperator{I}(ComplexF64(c))
scalarop(c::Number, V::ElementarySpace) = scalarop(c, sectortype(V))

# Inverse of `bondcharges` (multiplicity-free: all vertex labels 1). The 4-arg `FusionTree` form
# for K = 0: the 3-arg one throws for `MultipleFusion`.
function _tree_from_bonds(charges::AbstractVector{I}, bonds::AbstractVector{I}) where {I <: Sector}
    K = length(charges)
    K == 0 && return FusionTree{I}((), unit(I), (), ())
    unc = ntuple(i -> charges[i], K)
    isd = ntuple(_ -> false, K)
    inner = ntuple(i -> bonds[i + 1], max(0, K - 2))   # innerlines = bonds[2 : K-1]
    vtx = ntuple(_ -> 1, max(0, K - 1))
    return FusionTree{I}(unc, bonds[K], isd, inner, vtx)
end

"""
    Term{I<:Sector}

One term: `sites` (ascending, unique), one `ITOKey` each (letter, running bond charge, vertex label)
and a reduced `coeff`. Multiplicity-free fusion only. Never mutated, so terms share vectors;
`==`/`hash` ignore the coefficient, which is what makes summing coincident terms well defined.
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

arity(t::Term) = length(t.sites)

ops(t::Term{I}) where {I} = IrrepOperator{I}[k.op for k in t.keys]

# running bond charge after the last site
total(t::Term{I}) where {I} = isempty(t.keys) ? unit(I) : last(t.keys).bond

# the caterpillar coupling tree, rebuilt from the running bond charges
tree(t::Term{I}) where {I} = _tree_from_bonds(I[k.op.c for k in t.keys], I[k.bond for k in t.keys])

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

# A product acting more than once on one site (`a[i] * b[i]`), unevaluated until it enters an
# `OperatorSum`. Letters in written order with the running bond charges of the written-order
# caterpillar; the operator is `coeff` times that caterpillar's product.
struct PendingTerm{I <: Sector}
    sites::Vector{Int}
    keys::Vector{ITOKey{I}}
    coeff::ComplexF64
end

"""
    Terms{I<:Sector}

The compressible ITO operator: a lattice-free bag of [`Term`](@ref)s, returned by placement (`A[i]`),
[`couple`](@ref) and `dot`, and combined by `+ - * /`. Terms are appended as given;
[`canonicalize!`](@ref) normalises in place, and `length`, iteration, indexing and `≈` go through it.
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

# the appended count, before coincident terms are summed (tests)
nterms_raw(ts::Terms) = length(ts.terms)

# Everything that observes the term set normalises first.
Base.length(ts::Terms) = length(canonicalize!(ts).terms)
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

# `one` is the term with no active sites; `zero` is the empty bag
Base.one(::Union{Terms{I}, Type{Terms{I}}}) where {I} =
    Terms{I}(Term{I}[Term{I}(Int[], ITOKey{I}[], ComplexF64(1))])
Base.zero(::Union{Terms{I}, Type{Terms{I}}}) where {I} = Terms{I}()
Base.empty(::Terms{I}) where {I} = Terms{I}()

Base.copy(ts::Terms{I}) where {I} = Terms{I}(copy(ts.terms), copy(ts.pending))

# `≈`: close as sparse vectors over the canonical terms (a missing term counts as zero); `==` is exact.
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
    join(io, ("$(t.coeff) * [sites=$(t.sites), ops=$(ops(t)), total=$(total(t))]" for t in ts.terms), " + ")
    return print(io, ")")
end

_tolattice(lat::AbstractLattice) = lat
function _tolattice(sites)
    applicable(iterate, sites) ||
        throw(ArgumentError("a lattice must be a `FiniteChain`, an `InfiniteChain`, or an iterable of operator `Basis`es, got $(typeof(sites))"))
    return FiniteChain(vec(collect(sites)))
end

# Checks letters against the lattice's bases when terms enter an `OperatorSum`; no site range on an
# `InfiniteChain`.
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

Accumulate terms into one [`Terms`](@ref) in one pass. Each argument is a [`Term`](@ref), a `Terms` bag
or an arbitrarily nested iterable of those; `opsum(lat, terms...)` builds an [`OperatorSum`](@ref).
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

# The accumulator is `nothing` until the first term fixes `I`; after that the typed collector runs.
_collect_any(::Nothing, t::Term{I}) where {I} = Terms{I}(Term{I}[t])
_collect_any(::Nothing, ts::Terms) = copy(ts)
_collect_any(::Nothing, itr) = foldl(_collect_any, itr; init = nothing)
_collect_any(out::Terms{I}, a) where {I} = _collect_terms!(out, a, I)
# a bare space is iterable, so a vector of them would otherwise be read as a nest of terms
_collect_any(::Nothing, ::ElementarySpace) = _spaceaslattice()
_collect_terms!(::Terms{I}, ::ElementarySpace, ::Type{I}) where {I} = _spaceaslattice()
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

Append terms to `ts` in place; the arguments are as for [`opsum`](@ref).
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
    # the pass-through is the bare identity: a K = 0 term
    return Terms{I}(
        Term{I}[
            ispassthrough(l) ? Term{I}(Int[], ITOKey{I}[], c) : Term{I}([site], [ITOKey{I}(l, l.c, 1)], c)
                for (l, c) in pairs(O)
        ]
    )
end

# Operand of `dot`: a single-site charged term; returns `(site, op)`.
function _single_site_term(b::Terms, ctx)
    length(b) == 1 || throw(ArgumentError("$ctx must be a single-term operator"))
    t = only(b.terms)
    arity(t) == 1 || throw(ArgumentError("$ctx must be a single-site charged operator"))
    return only(t.sites), only(t.keys).op
end

# Whether a leg may be *inserted* left of others: `UniqueFusion` forces every invalidated running
# charge and makes a transposition one scalar `Rsymbol(x, y, x ⊗ y)`; `SymmetricBraiding` makes
# over/under irrelevant.
_canreorder(::Type{I}) where {I <: Sector} =
    FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding

# Fermion-odd: a fermionic sector whose twist is -1 (not SU(2)'s half-integer spin, which is bosonic).
_isfermionodd(c::Sector) = BraidingStyle(typeof(c)) isa Fermionic && real(twist(c)) < 0

# Extend every term of `a` by every single-site term of `b`, fusing to `target(running total, b's
# letter)` (`nothing` drops the pair). A leg right of all of `a` is appended; otherwise it is inserted
# at its site position (`_canreorder`), the running charges after it recomputed and one R-symbol
# picked up per leg it braids past (for fermions, the anticommutation sign). Under a non-abelian
# symmetry it is appended in written order and re-coupled into site order (`_recouple`), which may
# give several terms.
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

            # position of the first of `a`'s sites strictly right of `b`'s
            p = na + 1
            for j in 1:na
                s = ta.sites[j]
                s == sb && throw(ArgumentError("couple: operators must act on distinct sites"))
                s > sb && (p = j; break)
            end

            tgt = target(run, opb)
            tgt === nothing && continue
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
    couple(a::Terms, b::Terms, cs::Terms...; to = unit(I))

Left-nested (caterpillar) coupling of the composite `a` with the charged single-site operator `b`,
fusing their charges to `to` (default: the unit sector, a Hamiltonian term). The operands may be
written in any site order: an out-of-order leg picks up its braiding phase, so
`couple(cd[i + 1], c[i]) == -couple(c[i], cd[i + 1])`. Both operands may be composite; pairs that
cannot fuse to `to` are dropped, and it is an error if none do.

The variadic form takes every intermediate channel the charges force and errors where one is a
genuine choice (see [`couple_channels`](@ref)); nest `couple` to name it. For SU(2),
`couple(S[1], S[2], S[3])` is forced, `couple(S[1], S[2], S[3], S[4])` is not.
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
# later operand.
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

The intermediate coupling channels a caterpillar [`couple`](@ref) of these operands may use, one
tuple of `N = length(ops) - 1` charges each. The variadic `couple` needs no channel argument exactly
when this returns a single tuple.
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

The Cartesian scalar product of two single-site operators: the singlet `couple(a, b)` times
`-√dim(c)` (`Sᵢ·Sⱼ = -√3 [S⊗S]⁽⁰⁾`). Either site order is accepted; swapping costs the R-symbol
`Rsymbol(cₐ, c_b, unit)` (`-1` for odd fermionic or half-integer SU(2) charges). Unlike `couple`,
it does not distribute over composite operands.
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

    # `b` is left of `a`: swap into storage order with one scalar R-symbol (no F-move needed)
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
