# Unplaced multi-site operators. A `Term` is `(sites, keys, coeff)`, so until now a K-site operator
# could only exist *placed*; `project(h, sites)` had to take the sites and every model re-projected
# the same bond block once per bond. `LocalOperator{I}` fills that cell of the type grid: a `Terms`
# bag on the relative sites `1:K`, placed by `B[i]` / `B[s₁, …, s_K]` — a monotone relabelling of
# each term's `sites`, so the keys (letters, running bond charges, vertex labels) carry over
# untouched and `canonicalize!`, `≈`, `==` and `show` are reused unchanged.
#
# Every term occupies all `K` slots. A slot an operator does not act on — the scalar in
# `dot(S, S) + 1/4`, the `1/2` in `couple(Sz + 1/2, Sz)` — is held by the `passthrough` sentinel
# letter rather than by a materialised identity (whether a trivial-charge letter *is* the identity
# depends on the site's space, which an unplaced operator does not know). Its key carries the
# running bond charge unchanged, so the caterpillar tree is still read off the keys. Placement
# drops those keys, so a placed `Term` never contains one and nothing downstream changes; a term
# that is pass-through in every slot places as a `K = 0` scalar, exactly as `SiteOperator` placement
# does for its pass-through letter.
#
# Unplaced `couple`/`dot` lower their operands onto consecutive slots and run the *placed*
# implementation, so the forced-channel fold, `couple_channels` and every error message are the
# same code. Slot order is argument order, so this path never reorders legs: no R-symbol, no
# `_canreorder` gate.
#
# `SiteOperator` stays a separate type: it is the entry type of every bond matrix in the sweep and
# must remain a flat letter → coefficient map. "Site" means one site, "local" means K.
#
# Design note: `research/local-operators.md` §3–4. The lattice-carrying container (§5) is the next
# step.

using TensorKit: Sector, unit
using VectorInterface: VectorInterface, scale
using LinearAlgebra: LinearAlgebra

"""
    LocalOperator{I<:Sector}

An operator on `K` consecutive **slots**, not yet placed on the lattice: a [`Terms`](@ref) bag on the
relative sites `1:K`, every term occupying all `K` of them. Obtained from [`project`](@ref) (the
`project(h)` form), from [`couple`](@ref) or `dot` of unplaced operands (`SiteOperator`s and
`LocalOperator`s, slots concatenated in argument order), or by converting a [`SiteOperator`](@ref)
(`K = 1`); combined with ordinary arithmetic (`+ - * /` between operators of equal `K`, and `B + α`
for a scalar `α`).

Place it with indexing, which produces a `Terms` bag and is where the slots become lattice sites:

```julia
B = dot(S, S)                # K = 2, never placed
B[i]                         # contiguous: sites i:i+K-1
B[i, j]                      # explicit: K strictly increasing sites; the gap is passed through
```

Placement relabels the sites and nothing else — keys and coefficients are shared with `B` — so it is
valid for any symmetry, including fermionic ones: a site inside a gap is a pass-through, and the
sweep reconstructs the running bond charge (and with it the graded sign) crossing it.

A slot a term does not act on (the `1/4` in `dot(S, S) + 1/4`, the `1/2` in `couple(Sz + 1/2, Sz)`)
is a **pass-through slot**: it is held by the pass-through letter, which placement drops, so
`(dot(S, S) + 1/4)[i]` is `dot(S[i], S[i + 1]) + 1/4 · one(Terms{I})` — the same bag the placed
spelling gives.

`length(B)` is the number of canonical terms, as for `Terms`; `nsites(B)` is `K`, which the
operator remembers even when it has no terms (`zero(B)`).
"""
struct LocalOperator{I <: Sector}
    terms::Terms{I}
    nsites::Int
    function LocalOperator{I}(terms::Terms{I}, nsites::Integer) where {I <: Sector}
        K = Int(nsites)
        K >= 1 || throw(ArgumentError("a LocalOperator must act on at least one slot, got nsites = $K"))
        # Full support is the invariant everything here leans on: placement maps slot `k` to
        # `sites[k]`, which has no meaning for a term that skips a slot.
        for t in terms.terms
            t.sites == 1:K || throw(
                ArgumentError(
                    "every term of a LocalOperator on $K slots must act on exactly the slots 1:$K, " *
                        "got a term on sites $(t.sites)"
                )
            )
        end
        return new{I}(terms, K)
    end
end

# Constructors
# ------------
LocalOperator(terms::Terms{I}, nsites::Integer) where {I} = LocalOperator{I}(terms, nsites)

# `K` read off the terms; an empty bag cannot say how many slots it has.
function LocalOperator(terms::Terms{I}) where {I}
    isempty(terms.terms) && throw(
        ArgumentError("cannot infer the number of slots of an empty LocalOperator; pass `nsites`")
    )
    return LocalOperator{I}(terms, arity(first(terms.terms)))
end

"""
    LocalOperator(O::SiteOperator) -> LocalOperator

The one-slot operator with the letters of `O`. A pass-through (scalar) component of `O` becomes a
pass-through slot, so `LocalOperator(O)[i] == O[i]` letter for letter, scalar included.
"""
LocalOperator(O::SiteOperator{I}) where {I} = LocalOperator{I}(_slotterms(O, 0), 1)

sectortype(::Type{LocalOperator{I}}) where {I} = I
VectorInterface.scalartype(::Type{<:LocalOperator}) = ComplexF64

"""
    nsites(B::LocalOperator) -> Int
    nsites(O::SiteOperator) -> 1

The number of slots `K` of `B`, i.e. the number of sites every placement `B[…]` acts on.
"""
nsites(B::LocalOperator) = B.nsites
nsites(::SiteOperator) = 1

# Container interface: the canonical term set, exactly as `Terms` reports it.
Base.length(B::LocalOperator) = length(B.terms)
Base.isempty(B::LocalOperator) = isempty(B.terms)
Base.iszero(B::LocalOperator) = isempty(B.terms)
Base.iterate(B::LocalOperator, args...) = iterate(B.terms, args...)
Base.eltype(::Type{LocalOperator{I}}) where {I} = Term{I}

# Placement
# ---------
"""
    B[i]               -> Terms      # contiguous: sites i:i+K-1
    B[s₁, s₂, …, s_K]  -> Terms      # explicit, strictly increasing

Place a `K`-slot [`LocalOperator`](@ref) on the lattice. One index places it contiguously from site
`i`; `K` indices name every site, in strictly increasing order (for `K = 1` the two readings
coincide). Any other number of indices is an error.

Placement is the monotone relabelling `k ↦ sites[k]` of each term's sites: letters, running bond
charges and coefficients are unchanged, so it is exact for any symmetry. Sites inside a gap become
pass-throughs, which the sweep reconstructs — for a fermionic bond charge crossing the gap this is
what carries the graded (Jordan–Wigner) string, so `project(hop)[i, j]` is exactly the term
`couple(cd[i], c[j])` writes.
Non-monotone placement (`B[j, i]` with `j > i`) would need braiding data and is not supported.

A pass-through slot (see [`LocalOperator`](@ref)) is dropped on placement — its site simply does
not appear in the placed term — so a term that is pass-through in every slot places as the `K = 0`
scalar term, as `scalarop(α, I)[i]` does.

```jldoctest
julia> using TensorKit, LinearAlgebra

julia> V = SU2Space(1//2 => 1);

julia> B = project(OpSum.instantiate(dot(spin(V)[1], spin(V)[2]), [V, V]));

julia> B[3] ≈ dot(spin(V)[3], spin(V)[4])
true

julia> B[1, 4] ≈ dot(spin(V)[1], spin(V)[4])
true

julia> (dot(spin(V), spin(V)) + 1/4)[1] ≈ dot(spin(V)[1], spin(V)[2]) + one(Terms{SU2Irrep}) / 4
true
```
"""
function Base.getindex(B::LocalOperator{I}, ind::Integer, inds::Integer...) where {I}
    K = B.nsites
    n = 1 + length(inds)
    if n == 1
        first = Int(ind)
        first >= 1 || throw(ArgumentError("site index must be ≥ 1, got $first"))
        sites = collect(first:(first + K - 1))
    elseif n == K
        sites = Int[Int(ind), (Int(s) for s in inds)...]
        sites[1] >= 1 || throw(ArgumentError("site index must be ≥ 1, got $(sites[1])"))
        all(k -> sites[k] < sites[k + 1], 1:(K - 1)) || throw(
            ArgumentError("placement sites must be strictly increasing, got $sites")
        )
    else
        throw(
            ArgumentError(
                "a LocalOperator on $K slots is placed with one site index (contiguous) or exactly " *
                    "$K (explicit), got $n"
            )
        )
    end
    # Canonicalise once here rather than once per placed copy: the bag is the same operator either
    # way, and a translation-invariant model places the same block `N` times.
    ts = canonicalize!(B.terms).terms
    out = Vector{Term{I}}(undef, length(ts))
    for (m, t) in enumerate(ts)
        if any(_ispassthroughkey, t.keys)
            # A pass-through slot leaves no trace in the placed term. The surviving keys need no
            # change: the pass-through injected trivial charge, so each one's running bond charge
            # is already the charge out of the previous *active* site.
            keep = findall(!_ispassthroughkey, t.keys)
            out[m] = Term{I}(sites[keep], t.keys[keep], t.coeff)
        else
            # Terms are never mutated, so every placed term may share the one `sites` vector.
            out[m] = Term{I}(sites, t.keys, t.coeff)
        end
    end
    return Terms{I}(out)
end

_ispassthroughkey(k::ITOKey) = ispassthrough(k.op)

# Arithmetic
# ----------
@noinline function _slotmismatch(a::LocalOperator, b::LocalOperator)
    return throw(
        ArgumentError(
            "cannot combine a LocalOperator on $(nsites(a)) slots with one on $(nsites(b)); place " *
                "them first if they are meant to overlap"
        )
    )
end

function Base.:+(a::LocalOperator{I}, b::LocalOperator{I}) where {I}
    a.nsites == b.nsites || _slotmismatch(a, b)
    return LocalOperator{I}(a.terms + b.terms, a.nsites)
end
Base.:-(a::LocalOperator{I}, b::LocalOperator{I}) where {I} = a + (-b)
VectorInterface.scale(B::LocalOperator{I}, α::Number) where {I} =
    LocalOperator{I}(scale(B.terms, α), B.nsites)
Base.:*(α::Number, B::LocalOperator) = scale(B, α)
Base.:*(B::LocalOperator, α::Number) = scale(B, α)
Base.:/(B::LocalOperator, α::Number) = scale(B, inv(α))
Base.:-(B::LocalOperator) = scale(B, -1)

# The all-pass-through term on `K` slots: `α` times the identity, before placement. Every running
# bond charge is the unit sector, so `unit(I)` is the right bond on every key here (unlike a
# pass-through slot *inside* a charged caterpillar, whose bond `_couple_terms` computes).
function _passthroughterm(::Type{I}, K::Int, α::Number) where {I <: Sector}
    key = ITOKey{I}(passthrough(I), unit(I), 1)
    return Term{I}(collect(1:K), fill(key, K), ComplexF64(α))
end

"""
    one(B::LocalOperator)
    B + α, α + B, B - α, α - B

The identity on the `nsites(B)` slots of `B`, as the single term that is pass-through in every slot
— it places as the `K = 0` scalar term, so `(B + α)[i] == B[i] + α * one(Terms{I})`. Adding a
`Number` to a `LocalOperator` adds that multiple of it.
"""
Base.one(B::LocalOperator{I}) where {I} =
    LocalOperator{I}(Terms{I}(Term{I}[_passthroughterm(I, B.nsites, 1)]), B.nsites)
Base.:+(B::LocalOperator{I}, α::Number) where {I} =
    LocalOperator{I}(B.terms + _passthroughterm(I, B.nsites, α), B.nsites)
Base.:+(α::Number, B::LocalOperator) = B + α
Base.:-(B::LocalOperator, α::Number) = B + (-α)
Base.:-(α::Number, B::LocalOperator) = (-B) + α

"""
    zero(B::LocalOperator)

The operator with no terms on the same `nsites(B)` slots. (There is no `zero(::Type)`: an empty
operator still has to know its `K`.)
"""
Base.zero(B::LocalOperator{I}) where {I} = LocalOperator{I}(Terms{I}(), B.nsites)

"""
    copy(B::LocalOperator) -> LocalOperator

A copy owning its own term vector, so `canonicalize!` on either side leaves the other alone; the
[`Term`](@ref)s themselves are shared, as for [`copy(::Terms)`](@ref).
"""
Base.copy(B::LocalOperator{I}) where {I} = LocalOperator{I}(copy(B.terms), B.nsites)

Base.:(==)(a::LocalOperator{I}, b::LocalOperator{I}) where {I} =
    a.nsites == b.nsites && a.terms == b.terms
Base.isapprox(a::LocalOperator{I}, b::LocalOperator{I}; kwargs...) where {I} =
    a.nsites == b.nsites && isapprox(a.terms, b.terms; kwargs...)

function Base.show(io::IO, B::LocalOperator{I}) where {I}
    canonicalize!(B.terms)
    print(io, "LocalOperator{", I, "}(nsites=", B.nsites, ": ")
    if isempty(B.terms.terms)
        print(io, "0")
    else
        join(io, ("$(t.coeff) * $(_slotbody(t))" for t in B.terms.terms), " + ")
    end
    return print(io, ")")
end
_slotbody(t::Term) = string("[ops=", ops(t), ", total=", total(t), "]")

# Unplaced coupling
# -----------------
# `couple`, `couple_channels` and `dot` on `SiteOperator`/`LocalOperator` operands. Each operand is
# lowered onto consecutive *slots* — operand `k` on `offset + 1 : offset + K_k` — and handed to the
# placed implementation, whose result is wrapped back up with `K = Σ K_k`. The lowered forms keep
# pass-through letters as keys (a slot must stay occupied), which is the one way they differ from
# placement.
#
# Slot order is argument order, so a later operand always acts to the right of everything before it
# and `_couple_terms` only ever appends: no leg is reordered, no R-symbol enters. That is also why
# the pass-through key's bond comes out right without a special case — `_couple_terms` writes
# `target(run, passthrough)`, which by `Nsymbol(run, unit, tot)` is `run` itself.

const UnplacedOperator{I} = Union{SiteOperator{I}, LocalOperator{I}}
const PlacedOperator{I} = Union{Terms{I}, Term{I}}

# `O` on slot `offset + 1`, every letter a term — the pass-through one too. Its key's bond is the
# running charge *before* the slot: `unit(I)` for a lone first operand (which is all this method is
# trusted for; a later operand's key is rewritten by `_couple_terms`, which reads only its letter).
function _slotterms(O::SiteOperator{I}, offset::Int) where {I}
    out = Term{I}[]
    sizehint!(out, length(O))
    site = Int[offset + 1]
    for (letter, coeff) in pairs(O)
        push!(out, Term{I}(site, ITOKey{I}[ITOKey{I}(letter, letter.c, 1)], ComplexF64(coeff)))
    end
    return Terms{I}(out)
end

# `B` shifted onto slots `offset + 1 : offset + K`; keys shared, pass-throughs kept.
function _slotterms(B::LocalOperator{I}, offset::Int) where {I}
    ts = canonicalize!(B.terms).terms
    offset == 0 && return Terms{I}(copy(ts))
    sites = collect((offset + 1):(offset + B.nsites))
    return Terms{I}(Term{I}[Term{I}(sites, t.keys, t.coeff) for t in ts])
end

# The operands of an unplaced `couple`, lowered onto consecutive slots. `I` is bound through the
# first operand (a `Vararg` alone would leave it unbound on the empty tuple).
function _slotoperands(ops::Tuple{UnplacedOperator{I}, Vararg{UnplacedOperator{I}}}) where {I}
    out = Terms{I}[]
    offset = 0
    for (k, o) in enumerate(ops)
        # The caterpillar extends by one *letter* at a time. Fusing a whole block onto it would
        # couple two trees, which is `via`, and deferred — the placed form says the same thing in
        # terms of sites, so name the slot count here where it is visible.
        k > 1 && nsites(o) > 1 && throw(
            ArgumentError(
                "couple: operand $k acts on $(nsites(o)) slots, but every operand after the first " *
                    "must be a single-slot operator — coupling a multi-slot block onto a " *
                    "caterpillar is tree-structured (`via`) coupling, which is deferred. Couple " *
                    "the single-site operators it is made of instead."
            )
        )
        push!(out, _slotterms(o, offset))
        offset += nsites(o)
    end
    return out
end

"""
    couple(a, b, rest...; to = unit(I))   # a, b, rest… : SiteOperator or LocalOperator

The **unplaced** coupling: the same caterpillar as the placed form, on `nsites(a) + nsites(b) + …`
slots, returned as a [`LocalOperator`](@ref). Slots are concatenated in argument order — `b` acts
on the slot(s) right after `a`'s — so placing the result is the placed coupling of the placed
operands, and a block is written once and placed wherever it is needed:

```julia
hop = couple(cd, c)                       # K = 2, no site in sight
hop[i, j] ≈ couple(cd[i], c[j])           # for any i < j
```

`to`, the forced-channel fold of the variadic form, [`couple_channels`](@ref) and every error are
exactly those of the placed `couple`. Since slot order *is* argument order nothing is ever
reordered, so no braiding phase enters on this path, whatever the symmetry. The first operand may
have any number of slots; every later one must be a single-slot operator (fusing a whole block onto
the caterpillar is tree-structured coupling, which is deferred).

A pass-through (scalar) part of an operand occupies its slot as a pass-through slot:
`couple(Sz + 1/2, Sz)` has the terms `Sz ⊗ Sz` and `1/2 · 𝟙 ⊗ Sz`, and places as `couple(Sz[i],
Sz[j]) + Sz[j] / 2`.

Operands must be all unplaced or all placed; mixing the two is an error.

```jldoctest
julia> using TensorKit

julia> F = fermion_ops();

julia> hop = couple(F.cd, F.c);

julia> hop[1, 3] ≈ couple(F.cd[1], F.c[3])
true

julia> couple(F.cd, F.c, F.cd, F.c)[2] ≈ couple(F.cd[2], F.c[3], F.cd[4], F.c[5])
true
```
"""
function couple(
        a::UnplacedOperator{I}, b::UnplacedOperator{I}, rest::UnplacedOperator{I}...;
        to = unit(I), via = nothing
    ) where {I}
    ops = (a, b, rest...)
    out = couple(_slotoperands(ops)...; to, via)
    return LocalOperator{I}(out, sum(nsites, ops))
end

"""
    couple_channels(a, rest...; to = unit(I))   # a, rest… : SiteOperator or LocalOperator

The channel tuples the unplaced [`couple`](@ref) of these operands may use — the same list the
placed form returns for the placed operands, since only the charges enter.
"""
function couple_channels(a::UnplacedOperator{I}, rest::UnplacedOperator{I}...; to = unit(I)) where {I}
    isempty(rest) &&
        throw(ArgumentError("couple_channels: needs at least two operands"))
    return couple_channels(_slotoperands((a, rest...))...; to)
end

"""
    dot(a, b)   # a, b : SiteOperator or LocalOperator

The **unplaced** scalar product: `dot(a[1], b[2])` as a two-slot [`LocalOperator`](@ref), so that
`dot(S, S)[i, j] ≈ dot(S[i], S[j])` for every `i < j`. The Cartesian factor `-√dim(c)` is the
placed one's; since `b`'s slot follows `a`'s, the operands are never swapped and no braiding phase
enters. The operands must be single-letter, single-slot, as for the placed `dot`.
"""
function LinearAlgebra.dot(a::UnplacedOperator{I}, b::UnplacedOperator{I}) where {I}
    out = LinearAlgebra.dot(_slotterms(a, 0), _slotterms(b, nsites(a)))
    return LocalOperator{I}(out, nsites(a) + nsites(b))
end

# Mixed placed/unplaced operands. The method is wide enough to catch every mixture, so the body
# has to tell a mixture apart from a call no placed method takes either (all `Term`s, say), which
# keeps the latter a `MethodError` as before.
function couple(
        a::Union{PlacedOperator{I}, UnplacedOperator{I}},
        b::Union{PlacedOperator{I}, UnplacedOperator{I}},
        rest::Union{PlacedOperator{I}, UnplacedOperator{I}}...; kwargs...
    ) where {I}
    return _mixedoperands(couple, (a, b, rest...))
end
function couple_channels(
        a::Union{PlacedOperator{I}, UnplacedOperator{I}},
        rest::Union{PlacedOperator{I}, UnplacedOperator{I}}...; kwargs...
    ) where {I}
    return _mixedoperands(couple_channels, (a, rest...))
end
function LinearAlgebra.dot(
        a::Union{PlacedOperator{I}, UnplacedOperator{I}},
        b::Union{PlacedOperator{I}, UnplacedOperator{I}}
    ) where {I}
    return _mixedoperands(LinearAlgebra.dot, (a, b))
end

@noinline function _mixedoperands(f, ops::Tuple)
    unplaced = findall(o -> o isa UnplacedOperator, ops)
    placed = findall(o -> o isa PlacedOperator, ops)
    (isempty(unplaced) || isempty(placed)) && throw(MethodError(f, ops))
    return throw(
        ArgumentError(
            "$(nameof(f)): operand(s) $unplaced are unplaced (SiteOperator/LocalOperator) and " *
                "operand(s) $placed are placed (Terms/Term). Place everything — `A[i]` — to get a " *
                "`Terms` bag, or nothing to get a `LocalOperator` and place that."
        )
    )
end
