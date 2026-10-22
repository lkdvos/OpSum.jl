# Coupling of placed operators into one caterpillar: `couple`, its channel bookkeeping, and `·`.

using TensorKit
using TensorKit: Sector, unit, dim, Nsymbol, Rsymbol, FusionStyle, UniqueFusion,
    BraidingStyle, SymmetricBraiding
using LinearAlgebra: LinearAlgebra

function Base.getindex(O::SiteOperator{I}, ind::Integer, inds::Integer...) where {I}
    isempty(inds) ||
        throw(ArgumentError("multi-site placement of an on-site operator is not supported; use `couple`"))
    site = Int(ind)
    site >= 1 || throw(ArgumentError("site index must be ≥ 1, got $site"))
    # the pass-through is the bare identity: a K = 0 term
    return Terms{I}(
        Term{I}[
            ispassthrough(l) ? Term{I}(Int[], Factor{I}[], c) : Term{I}([site], [Factor{I}(l, l.c, 1)], c)
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
                keys_w = vcat(ta.keys, Factor{I}(opb, tot, 1))
                append!(out, _recouple(vcat(ta.sites, sb), keys_w, coeff))
                continue
            end
            sites = Vector{Int}(undef, na + 1)
            keys = Vector{Factor{I}}(undef, na + 1)
            for j in 1:(p - 1)
                sites[j] = ta.sites[j]
                keys[j] = ta.keys[j]
            end
            sites[p] = sb
            if p == na + 1
                keys[p] = Factor{I}(opb, tot, 1)
            else
                bond = p == 1 ? opb.c : only(ta.keys[p - 1].bond ⊗ opb.c)
                keys[p] = Factor{I}(opb, bond, 1)
                for j in p:na
                    op = ta.keys[j].op
                    coeff *= Rsymbol(op.c, opb.c, only(op.c ⊗ opb.c))
                    bond = only(bond ⊗ op.c)
                    sites[j + 1] = ta.sites[j]
                    keys[j + 1] = Factor{I}(op, bond, 1)
                end
                bond == tot ||
                    invariant_violation("reordered coupling reached $bond, not the total $tot")
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
