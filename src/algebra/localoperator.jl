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

The identity on its `K` slots is `one(B)`, and `B + α`, `B - α` add a multiple of it (`α` times the
pass-through in every slot). Placing the identity gives the scalar term, which instantiates to `α·id`.
Coupling needs a charged operator in every term, so the identity cannot be an operand of
[`couple`](@ref) or `dot`.
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
function LocalOperator(O::SiteOperator{I}) where {I}
    # every letter is a term, the pass-through one too
    terms = Term{I}[Term{I}([1], Factor{I}[Factor{I}(l, l.c)], c) for (l, c) in pairs(O)]
    return LocalOperator{I}(Terms{I}(terms), 1)
end

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
    out = TermBag{I}()
    for t in B.terms.terms
        keep = findall(k -> !ispassthrough(k.letter), t.factors)
        if increasing    # a pure relabelling
            push!(out.terms, Term{I}(sites[keep], t.factors[keep], t.coeff))
        else    # a product in slot order
            written!(out, sites[keep], t.factors[keep], t.coeff)
        end
    end
    return Terms{I}(out)
end

function Base.:+(a::LocalOperator{I}, b::LocalOperator{I}) where {I}
    a.nsites == b.nsites || throw(
        ArgumentError("cannot combine a LocalOperator on $(a.nsites) slots with one on $(b.nsites)")
    )
    return LocalOperator{I}(a.terms + b.terms, a.nsites)
end
VectorInterface.scale(B::LocalOperator{I}, α::Number) where {I} =
    LocalOperator{I}(scale(B.terms, α), B.nsites)

# `α` times the identity on `K` slots: the pass-through in every slot, which placement drops.
function identityterm(::Type{I}, K::Int, α::Number) where {I}
    key = Factor{I}(passthrough(I), unit(I))
    return Term{I}(collect(1:K), fill(key, K), ComplexF64(α))
end

"""
    one(B::LocalOperator)

The identity on the `nsites(B)` slots of `B`.
"""
Base.one(B::LocalOperator{I}) where {I} =
    LocalOperator{I}(Terms{I}(Term{I}[identityterm(I, B.nsites, 1)]), B.nsites)

"""
    B + α, α + B, B - α, α - B

Add a multiple of the identity (a pass-through in every slot) to a [`LocalOperator`](@ref).
"""
Base.:+(B::LocalOperator{I}, α::Number) where {I} =
    LocalOperator{I}(B.terms + identityterm(I, B.nsites, α), B.nsites)
Base.:+(α::Number, B::LocalOperator) = B + α
Base.:-(B::LocalOperator, α::Number) = B + (-α)
Base.:-(α::Number, B::LocalOperator) = (-B) + α

Base.isapprox(a::LocalOperator{I}, b::LocalOperator{I}; kwargs...) where {I} =
    a.nsites == b.nsites && isapprox(a.terms, b.terms; kwargs...)

# Unplaced coupling: operands are placed on consecutive slots (no reordering, no R-symbol) and handed to
# the placed implementation, so `couple(ops...)[sites]` is the placed `couple` of the placed operands,
# errors included; the result is filled back to all its slots with pass-throughs. Like placed `couple`,
# every term of an operand needs a charged letter: a scalar part is rejected.

const UnplacedOperator{I} = Union{SiteOperator{I}, LocalOperator{I}}

# `O` placed on the slots after the first `offset` ones.
_placed(O::SiteOperator, offset::Int) = O[offset + 1]
_placed(B::LocalOperator, offset::Int) = B[offset + 1]

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
        push!(out, _placed(o, offset))
        offset += nsites(o)
    end
    return out
end

# The placed terms of an unplaced coupling on slots `1:K`, back as a `LocalOperator`: the slots a term does
# not act on hold the pass-through, which carries the running bond charge.
function _fillslots(ts::Terms{I}, K::Int) where {I}
    out = Term{I}[]
    for t in ts.terms
        factors = Vector{Factor{I}}(undef, K)
        bond, j = unit(I), 1
        for s in 1:K
            if j <= arity(t) && t.sites[j] == s
                bond = t.factors[j].bond
                factors[s] = t.factors[j]
                j += 1
            else
                factors[s] = Factor{I}(passthrough(I), bond)
            end
        end
        push!(out, Term{I}(collect(1:K), factors, t.coeff))
    end
    return LocalOperator{I}(Terms{I}(out), K)
end

"""
    couple(a, b, rest...; to = unit(I))   # a, b, rest… : SiteOperator or LocalOperator

The unplaced coupling: the placed caterpillar on consecutive slots, returned as a
[`LocalOperator`](@ref), so `couple(cd, c)[i, j] ≈ couple(cd[i], c[j])`, errors included. Every
operand after the first must be single-slot; `to` and the variadic forced-channel form are those of the
placed `couple`. Every term of an operand needs a charged letter (a scalar part, such as `S + 1`, is an
`ArgumentError`, as in the placed form).
"""
function couple(
        a::UnplacedOperator{I}, b::UnplacedOperator{I}, rest::UnplacedOperator{I}...;
        to = unit(I)
    ) where {I}
    ops = (a, b, rest...)
    return _fillslots(couple(_slotoperands(ops)...; to), sum(nsites, ops))
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
Each operand must be one charged single-site term, as in the placed form.
"""
function LinearAlgebra.dot(a::UnplacedOperator{I}, b::UnplacedOperator{I}) where {I}
    out = LinearAlgebra.dot(_placed(a, 0), _placed(b, nsites(a)))
    return _fillslots(out, nsites(a) + nsites(b))
end
