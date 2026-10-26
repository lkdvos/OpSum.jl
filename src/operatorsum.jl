# The lattice-carrying container; terms are checked against the lattice as they enter.

using TensorKit: Sector

"""
    OperatorSum{I<:Sector, L<:AbstractLattice}

A Hamiltonian on a lattice: a [`Terms`](@ref) bag of finite-range terms plus an [`ExpSum`](@ref) of
decaying channels. Build it with `opsum(lat, terms...)` or fill an `OperatorSum(lat)` with
[`opsum!`](@ref). On an infinite chain terms must be charge-neutral with `K ≥ 1`.
"""
struct OperatorSum{I <: Sector, L <: AbstractLattice}
    lattice::L
    terms::Terms{I}
    channels::ExpSum{I}
end

function OperatorSum(lat::AbstractLattice)
    I = sectortype(lat)
    return OperatorSum{I, typeof(lat)}(lat, Terms{I}(), ExpSum{I}())
end
OperatorSum(sites::AbstractVector{<:Basis}) = OperatorSum(FiniteChain(sites))

Base.length(H::OperatorSum) = length(H.terms)
Base.isempty(H::OperatorSum) = isempty(H.terms) && isempty(H.channels)

function Base.copy(H::OperatorSum{I, L}) where {I, L}
    return OperatorSum{I, L}(H.lattice, copy(H.terms), copy(H.channels))
end

# exact: the lattice (by its bases), the terms and the channels
function Base.:(==)(a::OperatorSum{I}, b::OperatorSum{I}) where {I}
    return a.lattice == b.lattice && a.terms == b.terms && a.channels == b.channels
end
Base.hash(H::OperatorSum, h::UInt) = hash(H.channels, hash(H.terms, hash(H.lattice, hash(:OperatorSum, h))))

"""
    opsum!(H::OperatorSum, terms...) -> H

Add `Term`s, `Terms`, `ExpSum`s or (nested) iterables of those to `H` in place. Everything is checked
against the lattice before anything is added.
"""
function opsum!(H::OperatorSum{I}, args...) where {I}
    buf = TermBag{I}(ExpSum{I}())
    for a in args
        collect_terms!(buf, a, I)
    end
    terms = buf.terms
    # on-site products are resolved here, where the sites' bases are known
    isempty(buf.pending) || append!(terms, resolve(buf.pending, H.lattice))
    checkinsert(H.lattice, terms, buf.channels)
    merge_terms!(H.terms, terms)
    addchannels!(H.channels, buf.channels)
    return H
end

"""
    opsum(lat, terms...) -> OperatorSum
    opsum(sites::AbstractVector{<:Basis}, terms...) -> OperatorSum

Accumulate `terms` into a new [`OperatorSum`](@ref) on the lattice `lat` (or on the finite chain of
`sites`). The terms are `Term`s, `Terms`, `ExpSum`s or nested iterables of those, as for
[`opsum!`](@ref); [`Terms`](@ref)`(terms...)` collects the same arguments without a lattice (and
without `ExpSum`s, which a `Terms` cannot hold).
"""
opsum(lat::AbstractLattice, args...) = opsum!(OperatorSum(lat), args...)
opsum(sites::AbstractVector{<:Basis}, args...) = opsum!(OperatorSum(sites), args...)

rejection(::OperatorSum) = "cannot add an OperatorSum to a Terms or with opsum!; combine them with `+`"

# Checks letters against the lattice's bases when terms enter an `OperatorSum`; no site range on an
# `InfiniteChain`.
function checklattice(terms::AbstractVector{Term{I}}, lat::AbstractLattice) where {I}
    isempty(terms) && return nothing
    _checksector(I, lat)
    seen = Dict{eltype(lat), Dict{I, Int}}()
    for t in terms
        _checksupport!(seen, lat, t.sites, t.factors)
    end
    return nothing
end
checklattice(ts::Terms, lat::AbstractLattice) = checklattice(ts.terms, lat)

function _checksector(::Type{I}, lat::AbstractLattice) where {I}
    J = sectortype(lat)
    J === I || throw(
        ArgumentError("cannot compress an operator over $I on a lattice of sector type $J")
    )
    return nothing
end

function _checksupport!(seen, lat::AbstractLattice, sites, factors::Vector{Factor{I}}) where {I}
    N = maxsite(lat)
    for (s, key) in zip(sites, factors)
        (N === nothing || 1 <= s <= N) || throw(
            ArgumentError("a term acts on site $s, outside the lattice `1:$N`")
        )
        V = lat[s]
        info = get!(() -> alphabet(V, I), seen, V)
        op = key.letter
        nmax = get(info, op.c, 0)
        1 <= op.n <= nmax || throw(
            ArgumentError(
                "letter $op does not exist on site $s ($V): that site carries " *
                    "$nmax operator(s) of charge $(op.c)"
            )
        )
    end
    return nothing
end

function checkinsert(lat::AbstractLattice, terms::Vector{Term{I}}, chans::ExpSum{I}) where {I}
    checklattice(terms, lat)
    checkgenerating(lat, terms, chans)
    checkchannels(lat, chans)
    return nothing
end

# A channel puts its letters on sites it does not name (translates, string), which must share one basis.
function checkchannels(lat::AbstractLattice, chans::ExpSum{I}) where {I}
    for k in keys(chans.channels)
        letters = Letter{I}[l for l in k.string.letters if !ispassthrough(l)]
        lat isa FiniteChain && append!(letters, (f.letter for f in k.support[2]))
        isempty(letters) && continue
        allequal(lat.sites) || throw(ArgumentError(_mixedbases_message(lat)))
        info = alphabet(first(lat.sites), I)
        for op in letters
            1 <= op.n <= get(info, op.c, 0) || throw(
                ArgumentError(
                    "letter $op of a decaying channel does not exist on the lattice ($(first(lat.sites))): " *
                        "its sites carry $(get(info, op.c, 0)) operator(s) of charge $(op.c)"
                )
            )
        end
    end
    return nothing
end

_mixedbases_message(::FiniteChain) =
    "a decaying channel on a finite chain moves by one site, so its letters land on sites with " *
    "different bases, where they mean different operators; use one basis on every site, or an " *
    "InfiniteChain whose cell is the pattern"
_mixedbases_message(::InfiniteChain) =
    "a decaying channel with a string fills every site of its gap, so its letters land on sites " *
    "with different bases, where they mean different operators; use one basis on every site"

checkgenerating(::FiniteChain, terms, chans) = nothing
function checkgenerating(lat::InfiniteChain, terms::Vector{Term{I}}, chans::ExpSum{I}) where {I}
    for t in terms
        checkgenerator(t)
    end
    seen = Dict{eltype(lat), Dict{I, Int}}()
    for k in keys(chans.channels)
        checkgenerator(k)
        _checksector(I, lat)
        _checksupport!(seen, lat, k.support...)
    end
    return nothing
end

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
Base.:-(a::OperatorSum, b::Union{OperatorSum, Term, Terms}) = a + (-b)
Base.:-(a::Union{Term, Terms}, b::OperatorSum) = a + (-b)

function _samelattice(a::OperatorSum, b::OperatorSum, op)
    a.lattice == b.lattice || throw(
        ArgumentError(
            "cannot $op two OperatorSums on different lattices: $(a.lattice) and $(b.lattice)"
        )
    )
    return nothing
end

Base.isapprox(a::OperatorSum{I}, b::OperatorSum{I}; kwargs...) where {I} =
    a.lattice == b.lattice && isapprox(a.terms, b.terms; kwargs...) &&
    isapprox(a.channels, b.channels; kwargs...)
"""
    adjoint(H::OperatorSum)
    H'

The hermitian conjugate; every term must have total charge `unit(I)`, and decaying channels are not
supported.
"""
function Base.adjoint(H::OperatorSum{I, L}) where {I, L}
    isempty(H.channels) || throw(
        ArgumentError("adjoint: an OperatorSum with decaying channels has no adjoint yet")
    )
    return OperatorSum{I, L}(H.lattice, adjoint_terms(H.terms, H.lattice), ExpSum{I}())
end

"""
    instantiate(H::OperatorSum)

Materialize `H` on its finite lattice as a `TensorMap`; decaying channels are truncated to the chain.
"""
function instantiate(H::OperatorSum{I, <:FiniteChain}) where {I}
    isempty(H.lattice) && throw(ArgumentError("cannot instantiate over an empty lattice"))
    isempty(H) && throw(ArgumentError("cannot instantiate an empty operator"))
    return dense(chain_terms(H), H.lattice.sites)
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
