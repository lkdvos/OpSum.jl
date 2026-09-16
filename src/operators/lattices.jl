# The lattices an operator can be compressed over. A term bag is latticeless — nothing in the term
# algebra or in the sweep needs a physical space, and the sweep needs only the site *count* — so the
# lattice is supplied where the MPO is formed, and these are the two things that can be supplied.
#
# `FiniteChain` and `InfiniteChain` differ in exactly two observable ways: whether `getindex` wraps,
# and whether a site index outside `1:length` is an error. Everything downstream branches on the type
# rather than on a flag.

using TensorKit: TensorKit, ElementarySpace, Sector, Trivial, sectortype

"""
    AbstractLattice

Supertype of the lattices [`irrep_mpo`](@ref) accepts: [`FiniteChain`](@ref) and
[`InfiniteChain`](@ref). A lattice names the physical space of every site, which is what tensor
assembly, [`instantiate`](@ref) and letter validation need; the compression itself needs only
`length` (the number of sites, or the unit-cell period).
"""
abstract type AbstractLattice end

"""
    FiniteChain(spaces::AbstractVector{<:ElementarySpace})
    FiniteChain(V::ElementarySpace, N::Integer)

An open chain of `length(spaces)` sites, carrying one physical space per site. `getindex` does *not*
wrap: site `i` exists only for `i ∈ 1:length(chain)`, and a term reaching outside that range is an
error rather than a wrap-around.

`FiniteChain(V, N)` is the uniform chain, replacing `fill(V, N)`. A bare
`AbstractVector{<:ElementarySpace}` is accepted anywhere a `FiniteChain` is, and converted.
"""
struct FiniteChain{S <: ElementarySpace} <: AbstractLattice
    spaces::Vector{S}
end
FiniteChain(spaces::AbstractVector{S}) where {S <: ElementarySpace} = FiniteChain{S}(collect(spaces))
FiniteChain(V::S, N::Integer) where {S <: ElementarySpace} = FiniteChain{S}(fill(V, Int(N)))
FiniteChain(lat::FiniteChain) = lat

"""
    InfiniteChain(spaces::AbstractVector)
    InfiniteChain(V::ElementarySpace)

An infinite chain with a repeating unit cell of `L = length(spaces)` sites, carrying one physical
space per site of the cell. Site `i` of the infinite lattice has space `spaces[mod1(i, L)]`, for any
`i ∈ ℤ` — `getindex` wraps, so this is the lattice of the whole chain and not just of the cell.

An operator compressed over one is a *generating set*: the operator represented is
`Σ_{n ∈ ℤ} translate(H, n·L)`, so each translation class must appear exactly once (see
[`unitcell_terms`](@ref)). Its terms may reach past the cell — `dot(S[1], S[2])` on a one-site cell
names site 2 — which is exactly what the wrap-around `getindex` is for.
"""
struct InfiniteChain{S <: ElementarySpace} <: AbstractLattice
    spaces::Vector{S}
    function InfiniteChain(spaces::AbstractVector{S}) where {S <: ElementarySpace}
        isempty(spaces) &&
            throw(ArgumentError("an InfiniteChain needs at least one site per unit cell"))
        return new{S}(collect(spaces))
    end
end
InfiniteChain(V::ElementarySpace) = InfiniteChain([V])
InfiniteChain(lat::InfiniteChain) = lat

Base.length(lat::AbstractLattice) = length(lat.spaces)
Base.isempty(lat::AbstractLattice) = isempty(lat.spaces)
Base.eltype(::Type{FiniteChain{S}}) where {S} = S
Base.eltype(::Type{InfiniteChain{S}}) where {S} = S
Base.iterate(lat::AbstractLattice, args...) = iterate(lat.spaces, args...)
Base.firstindex(::AbstractLattice) = 1
Base.lastindex(lat::AbstractLattice) = length(lat)
Base.:(==)(a::AbstractLattice, b::AbstractLattice) =
    typeof(a) == typeof(b) && a.spaces == b.spaces

# The one behavioural difference: the finite chain has an edge, the infinite one does not.
Base.getindex(lat::FiniteChain, i::Integer) = lat.spaces[i]
Base.getindex(lat::InfiniteChain, i::Integer) = lat.spaces[mod1(Int(i), length(lat))]

TensorKit.sectortype(lat::AbstractLattice) = sectortype(eltype(lat.spaces))
_lattice_sectortype(lat::AbstractLattice) =
    isempty(lat.spaces) ? Trivial : sectortype(first(lat.spaces))

Base.show(io::IO, lat::FiniteChain) = print(io, "FiniteChain(", lat.spaces, ")")
Base.show(io::IO, lat::InfiniteChain) = print(io, "InfiniteChain(", lat.spaces, ")")

"""
    windowlattice(lat::InfiniteChain, N::Int) -> FiniteChain

The first `N` sites of the infinite lattice as a finite chain, for the unrolled window the infinite
sweep runs on.
"""
windowlattice(lat::InfiniteChain, N::Int) = FiniteChain([lat[i] for i in 1:N])

# `nothing` for a lattice that has no edge; the site range a term must respect otherwise.
_maxsite(lat::FiniteChain) = length(lat)
_maxsite(::InfiniteChain) = nothing
