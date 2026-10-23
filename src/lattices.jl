# The lattices an operator is compressed over: one operator `Basis` per site. They differ only in whether
# `getindex` wraps and whether a site outside `1:length` is an error.

using TensorKit: TensorKit, Trivial, sectortype

"""
    AbstractLattice

Supertype of [`FiniteChain`](@ref) and [`InfiniteChain`](@ref), the lattices an [`OperatorSum`](@ref)
can carry. A lattice names the operator [`Basis`](@ref) of every site.
"""
abstract type AbstractLattice end

"""
    FiniteChain(bases::AbstractVector{<:Basis})
    FiniteChain(B::Basis, N::Integer)

An open chain with one operator [`Basis`](@ref) per site (all of one symmetry); the letters of a term on
site `i` index into `bases[i]`. `getindex` does not wrap. `FiniteChain(B, N)` is the uniform chain; a bare
vector of bases is accepted anywhere a `FiniteChain` is.
"""
struct FiniteChain{B <: Basis} <: AbstractLattice
    sites::Vector{B}
    FiniteChain{B}(sites::Vector{B}) where {B <: Basis} = new{B}(_checksites(sites))
end
FiniteChain(sites::AbstractVector{<:Basis}) = (v = _sitevector(sites); FiniteChain{eltype(v)}(v))
FiniteChain(sites::AbstractVector) = _notbases(sites)
FiniteChain(V, N::Integer) = FiniteChain(fill(V, Int(N)))
FiniteChain(lat::FiniteChain) = lat

"""
    InfiniteChain(bases::AbstractVector{<:Basis})
    InfiniteChain(B::Basis)

An infinite chain with a repeating unit cell of `L = length(bases)` sites; site `i ∈ ℤ` has basis
`bases[mod1(i, L)]`. An operator on it is a generating set (see [`mpo`](@ref)); terms may reach past the
cell.
"""
struct InfiniteChain{B <: Basis} <: AbstractLattice
    sites::Vector{B}
    function InfiniteChain{B}(sites::Vector{B}) where {B <: Basis}
        isempty(sites) &&
            throw(ArgumentError("an InfiniteChain needs at least one site per unit cell"))
        return new{B}(_checksites(sites))
    end
end
InfiniteChain(sites::AbstractVector{<:Basis}) = (v = _sitevector(sites); InfiniteChain{eltype(v)}(v))
InfiniteChain(sites::AbstractVector) = _notbases(sites)
InfiniteChain(V) = InfiniteChain([V])
InfiniteChain(lat::InfiniteChain) = lat

_sitevector(sites) = isempty(sites) ? Vector{eltype(sites)}(sites) : collect(map(identity, sites))

_notbases(sites) = throw(
    ArgumentError("a lattice needs one operator `Basis` per site, got $(unique(typeof.(sites)))")
)

# all sites of one symmetry
function _checksites(sites::Vector)
    isempty(sites) && return sites
    I = sectortype(space(first(sites)))
    all(s -> sectortype(space(s)) === I, sites) || throw(
        ArgumentError(
            "all sites of a lattice must share one symmetry, got $(unique(sectortype.(space.(sites))))"
        )
    )
    return sites
end

Base.length(lat::AbstractLattice) = length(lat.sites)
Base.isempty(lat::AbstractLattice) = isempty(lat.sites)
Base.eltype(::Type{FiniteChain{B}}) where {B} = B
Base.eltype(::Type{InfiniteChain{B}}) where {B} = B
Base.iterate(lat::AbstractLattice, args...) = iterate(lat.sites, args...)
Base.firstindex(::AbstractLattice) = 1
Base.lastindex(lat::AbstractLattice) = length(lat)
Base.:(==)(a::AbstractLattice, b::AbstractLattice) =
    typeof(a) == typeof(b) && a.sites == b.sites

Base.getindex(lat::FiniteChain, i::Integer) = lat.sites[i]
Base.getindex(lat::InfiniteChain, i::Integer) = lat.sites[mod1(Int(i), length(lat))]

TensorKit.sectortype(lat::AbstractLattice) =
    isempty(lat.sites) ? Trivial : sectortype(space(first(lat.sites)))

Base.show(io::IO, lat::FiniteChain) = print(io, "FiniteChain(", lat.sites, ")")
Base.show(io::IO, lat::InfiniteChain) = print(io, "InfiniteChain(", lat.sites, ")")

# last valid site, `nothing` if unbounded
maxsite(lat::FiniteChain) = length(lat)
maxsite(::InfiniteChain) = nothing
