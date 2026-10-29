# Term algebra over letters: `Term` (one term) → `Terms` (the operator, always in normal form) and
# collecting terms into one (`Terms(args...)`). Latticeless. Coupling lives in couple.jl.

using TensorKit
using TensorKit: Sector, FusionTree, unit

# Inverse of `bondcharges` (multiplicity-free: all vertex labels are 1). The 4-arg `FusionTree` form
# for K = 0: the 3-arg one throws for `MultipleFusion`.
function tree_from_bonds(charges::AbstractVector{I}, bonds::AbstractVector{I}) where {I <: Sector}
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
and a reduced `coeff`. Multiplicity-free fusion only. Never mutated, so terms share vectors.
`==`/`hash` are struct-like (sites, factors and coefficient); the normal form sums terms of equal
`support` (sites and factors), which is also the key of `≈`.
"""
struct Term{I <: Sector}
    sites::Vector{Int}
    factors::Vector{Factor{I}}
    coeff::ComplexF64
    function Term{I}(sites::Vector{Int}, factors::Vector{Factor{I}}, coeff::ComplexF64) where {I}
        length(sites) == length(factors) || throw(
            ArgumentError("a term needs one factor per site, got $(length(sites)) site(s) and $(length(factors)) factor(s)")
        )
        issorted(sites; lt = <=) ||
            throw(ArgumentError("a term's sites must be ascending and unique, got $sites"))
        return new{I}(sites, factors, coeff)
    end
end
Term(sites::Vector{Int}, factors::Vector{Factor{I}}, coeff::Number) where {I} =
    Term{I}(sites, factors, ComplexF64(coeff))

TensorKit.sectortype(::Type{Term{I}}) where {I} = I

arity(t::Term) = length(t.sites)

letters(t::Term{I}) where {I} = Letter{I}[k.letter for k in t.factors]

# running bond charge after the last site
total(t::Term{I}) where {I} = isempty(t.factors) ? unit(I) : last(t.factors).bond

# the caterpillar coupling tree, rebuilt from the running bond charges
tree(t::Term{I}) where {I} = tree_from_bonds(I[k.letter.c for k in t.factors], I[k.bond for k in t.factors])

Base.:(==)(x::Term{I}, y::Term{I}) where {I} =
    x.sites == y.sites && x.factors == y.factors && x.coeff == y.coeff
# field-wise: the default struct hash goes through `objectid` and is markedly slower in the sweep
Base.hash(t::Term, h::UInt) = hash(t.coeff, hash(t.factors, hash(t.sites, hash(:Term, h))))

# What identifies a term up to its coefficient: the key of the normal form and of `≈`.
const Support{I} = Tuple{Vector{Int}, Vector{Factor{I}}}
support(t::Term{I}) where {I} = (t.sites, t.factors)::Support{I}
samesupport(x::Term{I}, y::Term{I}) where {I} = x.sites == y.sites && x.factors == y.factors

# Total order for the normal form; the factors distinguish letters *and* couplings.
function Base.isless(x::Term{I}, y::Term{I}) where {I}
    x.sites == y.sites || return isless(x.sites, y.sites)
    return isless(x.factors, y.factors)
end

VectorInterface.scale(t::Term{I}, α::Number) where {I} =
    Term{I}(t.sites, t.factors, t.coeff * ComplexF64(α))

function Base.show(io::IO, t::Term)
    return print(io, "Term(sites=", t.sites, ", ops=", letters(t), ", total=", total(t), ", coeff=", t.coeff, ")")
end

# A product acting more than once on one site (`a[i] * b[i]`), unevaluated until it enters an
# `OperatorSum`. Letters in written order with the running bond charges of the written-order
# caterpillar; the operator is `coeff` times that caterpillar's product.
struct PendingTerm{I <: Sector}
    sites::Vector{Int}
    factors::Vector{Factor{I}}
    coeff::ComplexF64
end

"""
    Terms{I<:Sector}

The compressible operator: a lattice-free sum of [`Term`](@ref)s, returned by placement (`A[i]`),
[`couple`](@ref) and `dot`, and combined by `+ - * /`. It is always in normal form: terms sorted,
coincident terms (equal sites and factors) summed, zero coefficients dropped; every constructor
and operation returns it so, and reading never changes it. The exception is a product acting twice
on one site (`a[i] * b[i]`), kept apart as a pending product until it enters an
[`OperatorSum`](@ref); such a `Terms` has no normal form, and `length`, iteration and `≈` throw.
"""
struct Terms{I <: Sector}
    terms::Vector{Term{I}}
    pending::Vector{PendingTerm{I}}
    # normalises `terms` in place: the vector is owned by the new `Terms`
    function Terms{I}(terms::Vector{Term{I}}, pending::Vector{PendingTerm{I}}) where {I <: Sector}
        return new{I}(normalize_terms!(terms), pending)
    end
end
Terms{I}(terms::Vector{Term{I}}) where {I <: Sector} = Terms{I}(terms, PendingTerm{I}[])
Terms(terms::Vector{Term{I}}) where {I} = Terms{I}(terms)
Terms(t::Term{I}) where {I} = Terms{I}(Term{I}[t])

TensorKit.sectortype(::Type{Terms{I}}) where {I} = I

# A raw, unordered accumulator of terms and pending products, normalised once by `Terms{I}(bag)`.
# `channels` is the slot for decaying channels (an `ExpSum` when collecting for an `OperatorSum`,
# `nothing` when collecting a `Terms`, which has none).
struct TermBag{I <: Sector, C}
    terms::Vector{Term{I}}
    pending::Vector{PendingTerm{I}}
    channels::C
end
TermBag{I}() where {I <: Sector} = TermBag{I, Nothing}(Term{I}[], PendingTerm{I}[], nothing)
TermBag{I}(channels) where {I <: Sector} = TermBag{I, typeof(channels)}(Term{I}[], PendingTerm{I}[], channels)
Terms{I}(bag::TermBag{I, Nothing}) where {I <: Sector} = Terms{I}(bag.terms, bag.pending)
Terms(bag::TermBag{I, Nothing}) where {I} = Terms{I}(bag)

# --- normal form ----------------------------------------------------------------------------------

# the three-way comparison of the term order (by support)
function cmpsupport(x::Term{I}, y::Term{I}) where {I}
    c = cmp(x.sites, y.sites)
    return iszero(c) ? cmp(x.factors, y.factors) : c
end

# Normal form in place: no zero coefficients, strictly ascending supports. Already-normal input costs one pass.
function normalize_terms!(terms::Vector{Term{I}}) where {I}
    any(t -> iszero(t.coeff), terms) && filter!(t -> !iszero(t.coeff), terms)
    (length(terms) <= 1 || all(i -> cmpsupport(terms[i - 1], terms[i]) < 0, 2:length(terms))) && return terms

    sort!(terms)
    out = 0
    @inbounds for i in eachindex(terms)
        t = terms[i]
        if out >= 1 && samesupport(terms[out], t)
            prev = terms[out]
            terms[out] = Term{I}(prev.sites, prev.factors, prev.coeff + t.coeff)
        else
            out += 1
            terms[out] = t
        end
    end
    resize!(terms, out)
    any(t -> iszero(t.coeff), terms) && filter!(t -> !iszero(t.coeff), terms)
    return terms
end

# The sum of two normal forms, a merge in `O(n + m)`.
function merge_terms(a::Vector{Term{I}}, b::Vector{Term{I}}) where {I}
    out = Term{I}[]
    sizehint!(out, length(a) + length(b))
    i = j = 1
    while i <= length(a) && j <= length(b)
        x, y = a[i], b[j]
        c = cmpsupport(x, y)
        if c < 0
            push!(out, x)
            i += 1
        elseif c > 0
            push!(out, y)
            j += 1
        else
            s = x.coeff + y.coeff
            iszero(s) || push!(out, Term{I}(x.sites, x.factors, s))
            i += 1
            j += 1
        end
    end
    append!(out, view(a, i:length(a)))
    append!(out, view(b, j:length(b)))
    return out
end

# Add `new` (owned, any order) into `ts` in place, keeping its normal form; for `OperatorSum` insertion and `append!`.
function merge_terms!(ts::Terms{I}, new::Vector{Term{I}}) where {I}
    isempty(new) && return ts
    merged = merge_terms(ts.terms, normalize_terms!(new))
    resize!(ts.terms, length(merged))
    copyto!(ts.terms, merged)
    return ts
end

# Observers of the normal form refuse a bag that still holds unevaluated on-site products.
function check_resolved(ts::Terms)
    isempty(ts.pending) || throw(
        ArgumentError(
            "this operator holds $(length(ts.pending)) unresolved on-site product(s) (`a[i] * b[i]`, " *
                "or a pair placed on one site), which are only resolved in the sites' bases when it " *
                "enters an OperatorSum: `opsum(lat, h)`. Until then it has no normal form, so its " *
                "length, terms and `≈` are not defined."
        )
    )
    return ts
end

Base.length(ts::Terms) = length(check_resolved(ts).terms)
Base.isempty(ts::Terms) = isempty(check_resolved(ts).terms)
Base.iterate(ts::Terms, args...) = iterate(check_resolved(ts).terms, args...)
Base.eltype(::Type{Terms{I}}) where {I} = Term{I}
Base.getindex(ts::Terms, i::Integer) = check_resolved(ts).terms[i]
Base.firstindex(::Terms) = 1
Base.lastindex(ts::Terms) = length(ts)

# Each `+` is a merge, so folding over M terms is quadratic; use `Terms(args...)`.
Base.:+(a::Terms{I}, b::Terms{I}) where {I} =
    Terms{I}(merge_terms(a.terms, b.terms), vcat(a.pending, b.pending))
Base.:+(a::Terms{I}, t::Term{I}) where {I} = Terms{I}(merge_terms(a.terms, [t]), copy(a.pending))
Base.:+(t::Term{I}, a::Terms{I}) where {I} = a + t
VectorInterface.scale(ts::Terms{I}, α::Number) where {I} = Terms{I}(
    Term{I}[scale(t, α) for t in ts.terms],
    PendingTerm{I}[PendingTerm{I}(p.sites, p.factors, p.coeff * ComplexF64(α)) for p in ts.pending]
)

# `one` is the term with no active sites; `zero` is the empty operator
Base.one(::Union{Terms{I}, Type{Terms{I}}}) where {I} =
    Terms{I}(Term{I}[Term{I}(Int[], Factor{I}[], ComplexF64(1))])
Base.zero(::Union{Terms{I}, Type{Terms{I}}}) where {I} = Terms{I}()
Base.empty(::Terms{I}) where {I} = Terms{I}()

Base.copy(ts::Terms{I}) where {I} = Terms{I}(copy(ts.terms), copy(ts.pending))

# `≈`: close as sparse vectors over the terms (a missing term counts as zero); `==` is exact.
Base.isapprox(a::Terms{I}, b::Terms{I}; kwargs...) where {I} =
    _termsapprox(check_resolved(a).terms, check_resolved(b).terms; kwargs...)
Base.:(==)(a::Terms{I}, b::Terms{I}) where {I} = check_resolved(a).terms == check_resolved(b).terms
Base.hash(ts::Terms, h::UInt) = hash(check_resolved(ts).terms, hash(:Terms, h))

function Base.show(io::IO, ts::Terms{I}) where {I}
    isempty(ts.pending) || return print(
        io, "Terms{", I, "}(", length(ts.terms), " term(s) and ", length(ts.pending),
        " unresolved on-site product(s); add it to an OperatorSum to resolve them)"
    )
    print(io, "Terms(")
    join(io, ("$(t.coeff) * [sites=$(t.sites), ops=$(letters(t)), total=$(total(t))]" for t in ts.terms), " + ")
    return print(io, ")")
end

# `isapprox` of the two operators as sparse vectors over terms: a merge walk over the two sorted
# lists, matching terms by support.
function _termsapprox(
        ta::Vector{Term{I}}, tb::Vector{Term{I}}; atol::Real = 0, rtol::Real = Base.rtoldefault(Float64)
    ) where {I}
    d2 = 0.0
    i = j = 1
    while i <= length(ta) && j <= length(tb)
        x, y = ta[i], tb[j]
        if samesupport(x, y)
            d2 += abs2(x.coeff - y.coeff)
            i += 1
            j += 1
        elseif isless(x, y)
            d2 += abs2(x.coeff)
            i += 1
        else
            d2 += abs2(y.coeff)
            j += 1
        end
    end
    for k in i:length(ta)
        d2 += abs2(ta[k].coeff)
    end
    for k in j:length(tb)
        d2 += abs2(tb[k].coeff)
    end
    nrm(ts) = sqrt(sum(t -> abs2(t.coeff), ts; init = 0.0))
    return sqrt(d2) <= max(atol, rtol * max(nrm(ta), nrm(tb)))
end

"""
    Terms(args...)
    Terms{I}(args...)

Collect terms into one [`Terms`](@ref) in one pass (an `O(M log M)` sort, where folding with `+` is
quadratic). Each argument is a [`Term`](@ref), a `Terms` or an arbitrarily nested iterable of those;
`Terms(args...)` infers the sector `I` from the first term, `Terms{I}(args...)` states it (and
allows no arguments: `Terms{I}()` is the empty operator). A `Vector{Term{I}}` is taken over as is.
`opsum(lat, args...)` collects the same arguments into an [`OperatorSum`](@ref).
"""
function Terms(args...)
    isempty(args) && throw(
        ArgumentError(
            "Terms() has no terms, so the symmetry sector cannot be inferred; write `Terms{I}()` " *
                "for the empty operator"
        )
    )
    out = nothing
    for a in args
        out = collect_any(out, a)
    end
    out === nothing && throw(
        ArgumentError(
            "Terms: every argument was empty, so the symmetry sector cannot be inferred; write " *
                "`Terms{I}()` for the empty operator"
        )
    )
    return Terms(out)
end

function Terms{I}(args...) where {I <: Sector}
    bag = TermBag{I}()
    for a in args
        collect_terms!(bag, a, I)
    end
    return Terms{I}(bag)
end

# The accumulator is `nothing` until the first term fixes `I`; after that the typed collector runs.
collect_any(::Nothing, t::Term{I}) where {I} = collect_terms!(TermBag{I}(), t, I)
collect_any(::Nothing, ts::Terms{I}) where {I} = collect_terms!(TermBag{I}(), ts, I)
function collect_any(::Nothing, itr)
    check_collectable(itr)
    return foldl(collect_any, itr; init = nothing)
end
collect_any(out::TermBag{I}, a) where {I} = collect_terms!(out, a, I)

collect_terms!(out::TermBag{I}, t::Term{I}, ::Type{I}) where {I} = (push!(out.terms, t); out)
function collect_terms!(out::TermBag{I}, ts::Terms{I}, ::Type{I}) where {I}
    append!(out.terms, ts.terms)
    append!(out.pending, ts.pending)
    return out
end
function collect_terms!(out::TermBag{I}, itr, ::Type{I}) where {I}
    check_collectable(itr)
    for a in itr
        collect_terms!(out, a, I)
    end
    return out
end
# A term over a different symmetry than the one already accumulated is a mistake worth naming.
collect_terms!(out::TermBag{I}, ::Term{J}, ::Type{I}) where {I, J} = wrongsector(out, J)
collect_terms!(out::TermBag{I}, ::Terms{J}, ::Type{I}) where {I, J} = wrongsector(out, J)
wrongsector(::TermBag{I, Nothing}, ::Type{J}) where {I, J} = throw(
    ArgumentError("cannot combine an operator over $J with one over $I")
)
wrongsector(::TermBag{I}, ::Type{J}) where {I, J} = throw(
    ArgumentError("cannot add an operator over $J to an OperatorSum on a lattice of sector type $I")
)

# What no collector takes, with the reason (`nothing`: not rejected); types defined later add methods.
# A number or a string is iterable (of itself), so it would recurse forever.
rejection(::Any) = nothing
rejection(a::Number) = "cannot add the number $a; scale an operator instead"

function check_collectable(a)
    reason = rejection(a)
    reason === nothing || throw(ArgumentError(reason))
    applicable(iterate, a) || throw(
        ArgumentError(
            "cannot add a $(typeof(a)): expected a Term, Terms, ExpSum (`opsum` only) or an " *
                "iterable of those"
        )
    )
    return nothing
end

"""
    append!(ts::Terms, terms...) -> ts

Append terms to `ts` in place, keeping its normal form; the arguments are as for [`Terms`](@ref).
"""
function Base.append!(ts::Terms{I}, args...) where {I}
    bag = TermBag{I}()
    for a in args
        collect_terms!(bag, a, I)
    end
    append!(ts.pending, bag.pending)
    return merge_terms!(ts, bag.terms)
end
