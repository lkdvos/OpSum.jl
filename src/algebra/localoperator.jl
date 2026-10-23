# Unplaced K-site operators: a `Terms` bag on the relative sites `1:K`, placed by relabelling each
# term's sites.

using TensorKit: Sector, unit
using VectorInterface: VectorInterface, scale
using LinearAlgebra: LinearAlgebra

"""
    LocalOperator{I<:Sector}

An operator on `K` consecutive slots, not yet placed on the lattice: a [`Terms`](@ref) bag on the
sites `1:K`, every term occupying all of them (a slot it does not act on holds the pass-through
letter). Obtained from `project(h, B)`, from unplaced [`couple`](@ref)/`dot`, or from a
[`SiteOperator`](@ref). Place it with `B[i]` (sites `i:i+K-1`) or `B[s₁, …, s_K]`.
"""
struct LocalOperator{I <: Sector}
    terms::Terms{I}
    nsites::Int
    function LocalOperator{I}(terms::Terms{I}, nsites::Integer) where {I <: Sector}
        K = Int(nsites)
        K >= 1 || throw(ArgumentError("a LocalOperator must act on at least one slot, got nsites = $K"))
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

LocalOperator(terms::Terms{I}, nsites::Integer) where {I} = LocalOperator{I}(terms, nsites)

"""
    LocalOperator(O::SiteOperator)

The one-slot operator with the letters of `O`.
"""
LocalOperator(O::SiteOperator{I}) where {I} = LocalOperator{I}(_slotterms(O, 0), 1)

"""
    nsites(B::LocalOperator) -> Int

The number of slots `K` of `B`.
"""
nsites(B::LocalOperator) = B.nsites
nsites(::SiteOperator) = 1

"""
    B[i]               -> Terms      # contiguous: sites i:i+K-1
    B[s₁, s₂, …, s_K]  -> Terms      # explicit

Place a `K`-slot [`LocalOperator`](@ref) on the lattice: slot `k` acts on site `sₖ`. On strictly
increasing sites this only relabels, so it is exact for any symmetry (gap sites are pass-throughs).
Other placements are products in slot order (see `*`): out-of-order sites are reordered, and a repeated
site, `hop[i, i]`, is an on-site product kept unevaluated until it enters an [`OperatorSum`](@ref).
"""
function Base.getindex(B::LocalOperator{I}, ind::Integer, inds::Integer...) where {I}
    K = B.nsites
    n = 1 + length(inds)
    n in (1, K) || throw(
        ArgumentError("a LocalOperator on $K slots is placed with one site index or exactly $K, got $n")
    )
    sites = n == 1 ? collect(Int(ind):(Int(ind) + K - 1)) : Int[Int(ind), (Int(s) for s in inds)...]
    minimum(sites) >= 1 || throw(ArgumentError("site indices must be ≥ 1, got $sites"))
    increasing = issorted(sites; lt = <=)    # strictly increasing
    out = Terms{I}()
    for t in canonicalize!(B.terms).terms
        keep = findall(k -> !ispassthrough(k.letter), t.keys)
        if increasing    # a pure relabelling
            push!(out.terms, Term{I}(sites[keep], t.keys[keep], t.coeff))
        else    # a product in slot order
            _written!(out, sites[keep], t.keys[keep], t.coeff)
        end
    end
    return out
end

function Base.:+(a::LocalOperator{I}, b::LocalOperator{I}) where {I}
    a.nsites == b.nsites || throw(
        ArgumentError("cannot combine a LocalOperator on $(a.nsites) slots with one on $(b.nsites)")
    )
    return LocalOperator{I}(a.terms + b.terms, a.nsites)
end
VectorInterface.scale(B::LocalOperator{I}, α::Number) where {I} =
    LocalOperator{I}(scale(B.terms, α), B.nsites)

"""
    B + α, α + B, B - α, α - B

Add a multiple of the identity (a pass-through in every slot) to a [`LocalOperator`](@ref).
"""
function Base.:+(B::LocalOperator{I}, α::Number) where {I}
    key = Factor{I}(passthrough(I), unit(I))    # α on every slot, placed as the `K = 0` scalar
    return LocalOperator{I}(B.terms + Term{I}(collect(1:B.nsites), fill(key, B.nsites), ComplexF64(α)), B.nsites)
end
Base.:+(α::Number, B::LocalOperator) = B + α
Base.:-(B::LocalOperator, α::Number) = B + (-α)
Base.:-(α::Number, B::LocalOperator) = (-B) + α

Base.isapprox(a::LocalOperator{I}, b::LocalOperator{I}; kwargs...) where {I} =
    a.nsites == b.nsites && isapprox(a.terms, b.terms; kwargs...)

# Unplaced coupling: operands are lowered onto consecutive slots (no reordering, no R-symbol) and
# handed to the placed implementation.

const UnplacedOperator{I} = Union{SiteOperator{I}, LocalOperator{I}}

# `O` on slot `offset + 1`, every letter a term, the pass-through one too.
function _slotterms(O::SiteOperator{I}, offset::Int) where {I}
    out = Term{I}[]
    site = Int[offset + 1]
    for (letter, coeff) in pairs(O)
        push!(out, Term{I}(site, Factor{I}[Factor{I}(letter, letter.c)], ComplexF64(coeff)))
    end
    return Terms{I}(out)
end

function _slotterms(B::LocalOperator{I}, offset::Int) where {I}
    ts = canonicalize!(B.terms).terms
    offset == 0 && return Terms{I}(copy(ts))
    sites = collect((offset + 1):(offset + B.nsites))
    return Terms{I}(Term{I}[Term{I}(sites, t.keys, t.coeff) for t in ts])
end

function _slotoperands(ops::Tuple{UnplacedOperator{I}, Vararg{UnplacedOperator{I}}}) where {I}
    out = Terms{I}[]
    offset = 0
    for (k, o) in enumerate(ops)
        k > 1 && nsites(o) > 1 && throw(
            ArgumentError(
                "couple: operand $k acts on $(nsites(o)) slots; every operand after the first " *
                    "must be single-slot (coupling a block onto a caterpillar is deferred)"
            )
        )
        push!(out, _slotterms(o, offset))
        offset += nsites(o)
    end
    return out
end

"""
    couple(a, b, rest...; to = unit(I))   # a, b, rest… : SiteOperator or LocalOperator

The unplaced coupling: the placed caterpillar on consecutive slots, returned as a
[`LocalOperator`](@ref), so `couple(cd, c)[i, j] ≈ couple(cd[i], c[j])`. A pass-through (scalar)
part of an operand occupies its slot. Every operand after the first must be single-slot; `to` and
the variadic forced-channel form are those of the placed `couple`.
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
    couple_channels(a, rest...; to = unit(I))

The channel tuples the unplaced [`couple`](@ref) of these operands may use.
"""
function couple_channels(a::UnplacedOperator{I}, rest::UnplacedOperator{I}...; to = unit(I)) where {I}
    isempty(rest) && throw(ArgumentError("couple_channels: needs at least two operands"))
    return couple_channels(_slotoperands((a, rest...))...; to)
end

"""
    dot(a, b)   # a, b : SiteOperator or LocalOperator

The unplaced scalar product: a two-slot [`LocalOperator`](@ref) with `dot(S, S)[i, j] ≈ dot(S[i], S[j])`.
"""
function LinearAlgebra.dot(a::UnplacedOperator{I}, b::UnplacedOperator{I}) where {I}
    out = LinearAlgebra.dot(_slotterms(a, 0), _slotterms(b, nsites(a)))
    return LocalOperator{I}(out, nsites(a) + nsites(b))
end
