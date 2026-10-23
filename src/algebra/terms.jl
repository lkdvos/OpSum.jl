# Term algebra over letters: `Term` (one term) → `Terms` (a bag, the operator), its normal form
# (`canonicalize!`), and collecting terms into a bag (`opsum`). Latticeless. Coupling lives in couple.jl.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, unit
import TensorKit: sectortype

# Inverse of `bondcharges` (multiplicity-free: all vertex labels are 1). The 4-arg `FusionTree` form
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

One term: `sites` (ascending, unique), one `Factor` each (letter, running bond charge)
and a reduced `coeff`. Multiplicity-free fusion only. Never mutated, so terms share vectors;
`==`/`hash` ignore the coefficient, which is what makes summing coincident terms well defined.
"""
struct Term{I <: Sector}
    sites::Vector{Int}
    keys::Vector{Factor{I}}
    coeff::ComplexF64
    function Term{I}(sites::Vector{Int}, keys::Vector{Factor{I}}, coeff::ComplexF64) where {I}
        @assert length(sites) == length(keys) "sites/keys arity mismatch"
        @assert issorted(sites) && allunique(sites) "a term's sites must be ascending and unique"
        return new{I}(sites, keys, coeff)
    end
end
Term(sites::Vector{Int}, keys::Vector{Factor{I}}, coeff::Number) where {I} =
    Term{I}(sites, keys, ComplexF64(coeff))

sectortype(::Type{Term{I}}) where {I} = I

arity(t::Term) = length(t.sites)

ops(t::Term{I}) where {I} = Letter{I}[k.letter for k in t.keys]

# running bond charge after the last site
total(t::Term{I}) where {I} = isempty(t.keys) ? unit(I) : last(t.keys).bond

# the caterpillar coupling tree, rebuilt from the running bond charges
tree(t::Term{I}) where {I} = _tree_from_bonds(I[k.letter.c for k in t.keys], I[k.bond for k in t.keys])

Base.:(==)(x::Term{I}, y::Term{I}) where {I} = x.sites == y.sites && x.keys == y.keys
Base.hash(t::Term, h::UInt) = hash(t.keys, hash(t.sites, hash(:Term, h)))

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
    keys::Vector{Factor{I}}
    coeff::ComplexF64
end

"""
    Terms{I<:Sector}

The compressible operator: a lattice-free bag of [`Term`](@ref)s, returned by placement (`A[i]`),
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
    Terms{I}(Term{I}[Term{I}(Int[], Factor{I}[], ComplexF64(1))])
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

"""
    canonicalize!(ts::Terms) -> ts

Put `ts` in normal form in place: sort, sum coincident terms, drop cancelled ones. Idempotent.
"""
function canonicalize!(ts::Terms)
    isempty(ts.pending) || throw(
        ArgumentError(
            "this operator holds $(length(ts.pending)) unresolved on-site product(s) (`a[i] * b[i]`, " *
                "or a pair placed on one site), which are only resolved in the sites' bases when it " *
                "enters an OperatorSum: `opsum(lat, h)`. Until then it has no normal form, so its " *
                "length, terms and `≈` are not defined."
        )
    )
    _canonicalize!(ts.terms)
    return ts
end

# sort, merge adjacent equals, drop cancellations
function _canonicalize!(terms::Vector{Term{I}}) where {I}
    if length(terms) <= 1
        length(terms) == 1 && iszero(only(terms).coeff) && empty!(terms)   # so `isempty` cannot lie
        return terms
    end

    sort!(terms)
    out = 0
    @inbounds for i in eachindex(terms)
        t = terms[i]
        if out >= 1 && terms[out] == t
            prev = terms[out]
            terms[out] = Term{I}(prev.sites, prev.keys, prev.coeff + t.coeff)
        else
            out += 1
            terms[out] = t
        end
    end
    resize!(terms, out)

    any(t -> iszero(t.coeff), terms) && filter!(t -> !iszero(t.coeff), terms)
    return terms
end

# Both in normal form, so a positional walk is a set comparison.
function _termsequal(ta::Vector{Term{I}}, tb::Vector{Term{I}}) where {I}
    return length(ta) == length(tb) && all(((x, y),) -> x == y && x.coeff == y.coeff, zip(ta, tb))
end

# `isapprox` of the two operators as sparse vectors over terms.
function _termsapprox(
        ta::Vector{Term{I}}, tb::Vector{Term{I}}; atol::Real = 0, rtol::Real = Base.rtoldefault(Float64)
    ) where {I}
    diff = Dict{Term{I}, ComplexF64}(t => t.coeff for t in ta)
    for t in tb
        diff[t] = get(diff, t, zero(ComplexF64)) - t.coeff
    end
    nrm(ts) = sqrt(sum(t -> abs2(t.coeff), ts; init = 0.0))
    return sqrt(sum(abs2, values(diff); init = 0.0)) <= max(atol, rtol * max(nrm(ta), nrm(tb)))
end

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
