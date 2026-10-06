# The lattice-carrying container: a lattice, finite-range terms placed on it, and exponentially
# decaying channels. Letters are checked against the spaces as terms enter. Design note:
# `research/local-operators.md`.

using Dictionaries: Dictionary, setwith!
using TensorKit: Sector

"""
    OperatorSum{I<:Sector, L<:AbstractLattice}

A Hamiltonian on a lattice: the lattice, a [`Terms`](@ref) bag of finite-range terms and an
[`ExpSum`](@ref) of decaying channels. Build it with `opsum(lat, terms...)` or fill an
`OperatorSum(lat)` with [`opsum!`](@ref); `irrep_mpo(H)`, `islossless(H)`, `instantiate(H)` and
`jordan_mpo_tensors(H)` take it. Terms are checked against the lattice as they enter. On an
infinite chain `H` is a generating set, so terms must be charge-neutral and have `K ≥ 1`.
"""
struct OperatorSum{I <: Sector, L <: AbstractLattice}
    lattice::L
    terms::Terms{I}
    channels::ExpSum{I}
end

function OperatorSum(lat::AbstractLattice)
    I = _lattice_sectortype(lat)
    return OperatorSum{I, typeof(lat)}(lat, Terms{I}(), ExpSum{I}())
end
OperatorSum(sites) = OperatorSum(_tolattice(sites))

Base.length(H::OperatorSum) = length(H.terms)
Base.isempty(H::OperatorSum) = isempty(H.terms) && isempty(H.channels)

function Base.copy(H::OperatorSum{I, L}) where {I, L}
    return OperatorSum{I, L}(
        H.lattice, copy(H.terms), ExpSum{I}(copy(H.channels.channels))
    )
end
# --- inserting terms ------------------------------------------------------------------------------

"""
    opsum!(H::OperatorSum, terms...) -> H

Add `Term`s, `Terms`, `ExpSum`s or (nested) iterables of those to `H` in place, in one pass. Checked
against the lattice before anything is added. (`H + x` copies `H`.)
"""
function opsum!(H::OperatorSum{I}, args...) where {I}
    terms = Term{I}[]
    chans = Dictionary{ExpKey{I}, ComplexF64}()
    for a in args
        _collect_op!(terms, chans, a, I)
    end
    _checkinsert(H.lattice, terms, chans)
    append!(H.terms.terms, terms)
    if !isempty(chans)
        for (k, v) in pairs(chans)
            setwith!(+, H.channels.channels, k, v)
        end
        filter!(!iszero, H.channels.channels)
    end
    return H
end

"""
    opsum(lat, terms...) -> OperatorSum

Accumulate `terms` into a new [`OperatorSum`](@ref) on `lat`.
"""
opsum(lat::AbstractLattice, args...) = opsum!(OperatorSum(lat), args...)
opsum(sites::AbstractVector{<:ElementarySpace}, args...) = opsum!(OperatorSum(sites), args...)

_collect_op!(out, ch, t::Term{I}, ::Type{I}) where {I} = push!(out, t)
_collect_op!(out, ch, ts::Terms{I}, ::Type{I}) where {I} = append!(out, ts.terms)
function _collect_op!(out, ch, es::ExpSum{I}, ::Type{I}) where {I}
    for (k, v) in pairs(es.channels)
        setwith!(+, ch, k, v)
    end
    return ch
end
_collect_op!(out, ch, a::Union{SiteOperator, LocalOperator}, ::Type) = throw(
    ArgumentError("cannot add an unplaced $(nameof(typeof(a))): place it first (`A[i]`, `B[i, j]`)")
)
_collect_op!(out, ch, ::OperatorSum, ::Type) = throw(
    ArgumentError("cannot add an OperatorSum with opsum!; combine them with `+`")
)
# a number is iterable, so it would recurse forever
_collect_op!(out, ch, a::Number, ::Type) = throw(
    ArgumentError("cannot add the number $a to an OperatorSum; scale an operator instead")
)
function _collect_op!(out, ch, itr, ::Type{I}) where {I}
    applicable(iterate, itr) || throw(
        ArgumentError(
            "cannot add a $(typeof(itr)) to an OperatorSum: expected a Term, Terms, ExpSum or an " *
                "iterable of those"
        )
    )
    for a in itr
        _collect_op!(out, ch, a, I)
    end
    return out
end
for T in (:Term, :Terms, :ExpSum)
    @eval _collect_op!(out, ch, ::$T{J}, ::Type{I}) where {I, J} = _wrongsector_op(I, J)
end
_wrongsector_op(I, J) = throw(
    ArgumentError("cannot add an operator over $J to an OperatorSum on a lattice of sector type $I")
)

function _checkinsert(lat::AbstractLattice, terms::Vector{Term{I}}, chans) where {I}
    _checklattice(terms, lat)
    _checkgenerating(lat, terms, chans)
    return nothing
end

_checkgenerating(::FiniteChain, terms, chans) = nothing
function _checkgenerating(lat::InfiniteChain, terms::Vector{Term{I}}, chans) where {I}
    for t in terms
        _checkgenerator(t)
    end
    for k in keys(chans)
        _checkgenerator(k)
        _checklattice([k.term], lat)
    end
    return nothing
end

# --- arithmetic -----------------------------------------------------------------------------------

const _Summand{I} = Union{Term{I}, Terms{I}, ExpSum{I}}

Base.:+(H::OperatorSum, x::_Summand) = opsum!(copy(H), x)
Base.:+(x::_Summand, H::OperatorSum) = opsum!(copy(H), x)
function Base.:+(a::OperatorSum, b::OperatorSum)
    _samelattice(a, b, "+")
    return opsum!(copy(a), b.terms, b.channels)
end

VectorInterface.scale(H::OperatorSum{I, L}, α::Number) where {I, L} =
    OperatorSum{I, L}(H.lattice, scale(H.terms, α), scale(H.channels, α))
Base.:*(α::Number, H::OperatorSum) = scale(H, α)
Base.:*(H::OperatorSum, α::Number) = scale(H, α)
Base.:/(H::OperatorSum, α::Number) = scale(H, inv(α))
Base.:-(H::OperatorSum) = scale(H, -1)
Base.:-(a::OperatorSum, b::OperatorSum) = a + (-b)

function _samelattice(a::OperatorSum, b::OperatorSum, op)
    a.lattice == b.lattice || throw(
        ArgumentError(
            "cannot $op two OperatorSums on different lattices: $(a.lattice) and $(b.lattice)"
        )
    )
    return nothing
end

# --- comparison, display --------------------------------------------------------------------------

function _channelsapprox(a::ExpSum, b::ExpSum; kwargs...)
    length(a.channels) == length(b.channels) || return false
    for (k, v) in pairs(a.channels)
        haskey(b.channels, k) || return false
        isapprox(v, b.channels[k]; kwargs...) || return false
    end
    return true
end

Base.isapprox(a::OperatorSum{I}, b::OperatorSum{I}; kwargs...) where {I} =
    a.lattice == b.lattice && isapprox(a.terms, b.terms; kwargs...) &&
    _channelsapprox(a.channels, b.channels; kwargs...)
# --- the operations that need the spaces ----------------------------------------------------------

"""
    adjoint(H::OperatorSum)
    H'

The hermitian conjugate (every term must have total charge `unit(I)`; no decaying channels yet).
It needs the spaces, which is why it lives here and a bare `Terms` has no `h'`.
"""
function Base.adjoint(H::OperatorSum{I, L}) where {I, L}
    isempty(H.channels) || throw(
        ArgumentError("adjoint: an OperatorSum with decaying channels has no adjoint yet")
    )
    return OperatorSum{I, L}(H.lattice, _adjoint_terms(H.terms, H.lattice), ExpSum{I}())
end

"""
    instantiate(H::OperatorSum)

Materialize `H` on its finite lattice into a `TensorMap` (the dense oracle); decaying channels are
truncated to the chain ([`chain_terms`](@ref)).
"""
function instantiate(H::OperatorSum{I, <:FiniteChain}) where {I}
    isempty(H.lattice) && throw(ArgumentError("cannot instantiate over an empty lattice"))
    isempty(H) && throw(ArgumentError("cannot instantiate an empty operator"))
    return _instantiate_terms(chain_terms(H), H.lattice.spaces)
end
instantiate(::OperatorSum{I, <:InfiniteChain}) where {I} = throw(
    ArgumentError("cannot instantiate an operator on an infinite chain")
)

"""
    chain_terms(H::OperatorSum) -> Terms

The term sum `H` stands for on a finite chain: its terms plus every translate of every channel.
"""
function chain_terms(H::OperatorSum{I, <:FiniteChain}) where {I}
    return H.terms + expand_channels(H.channels, 1, length(H.lattice))
end
