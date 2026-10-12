# The lattices an operator can be compressed over; they enter with the `OperatorSum`, as terms are
# latticeless. A site is described by its operator `Basis` (basis.jl), whose letters the terms on it index
# into. `FiniteChain` and `InfiniteChain` differ in whether `getindex` wraps and whether a site outside
# `1:length` is an error; downstream code branches on the type.

using TensorKit: TensorKit, Trivial, sectortype

"""
    AbstractLattice

Supertype of the lattices an [`OperatorSum`](@ref) can carry: [`FiniteChain`](@ref) and
[`InfiniteChain`](@ref). A lattice names the operator [`Basis`](@ref) of every site (needed by tensor
assembly, [`instantiate`](@ref) and letter validation); the compression itself needs only `length`.
"""
abstract type AbstractLattice end

"""
    FiniteChain(bases::AbstractVector{<:Basis})
    FiniteChain(B::Basis, N::Integer)

An open chain of `length(bases)` sites, carrying one operator [`Basis`](@ref) per site: the letters of a
term on site `i` index into `bases[i]`. Sites with different spaces are combined by listing their bases,
e.g. `vcat(Bf, Bs, Bf)`; all must share one symmetry. `getindex` does *not* wrap: a term reaching outside
`1:length(chain)` is an error.

`FiniteChain(B, N)` is the uniform chain. A bare vector of bases is accepted anywhere a `FiniteChain` is.
"""
struct FiniteChain{S} <: AbstractLattice
    sites::Vector{S}
    FiniteChain{S}(sites::Vector{S}) where {S} = new{S}(_checksites(sites))
end
FiniteChain(sites::AbstractVector) = (v = _sitevector(sites); FiniteChain{eltype(v)}(v))
FiniteChain(V, N::Integer) = FiniteChain(fill(V, Int(N)))
FiniteChain(lat::FiniteChain) = lat

"""
    InfiniteChain(bases::AbstractVector)
    InfiniteChain(B::Basis)

An infinite chain with a repeating unit cell of `L = length(bases)` sites, one operator [`Basis`](@ref) per
site. Site `i ∈ ℤ` has basis `bases[mod1(i, L)]` (`getindex` wraps).

An operator compressed over one is a *generating set*: it represents `Σ_{n ∈ ℤ} translate(H, n·L)`, so each
translation class must appear exactly once (see [`unitcell_terms`](@ref)). Terms may reach past the
cell: `SS[1, 2]` on a one-site cell names site 2.
"""
struct InfiniteChain{S} <: AbstractLattice
    sites::Vector{S}
    function InfiniteChain{S}(sites::Vector{S}) where {S}
        isempty(sites) &&
            throw(ArgumentError("an InfiniteChain needs at least one site per unit cell"))
        return new{S}(_checksites(sites))
    end
end
InfiniteChain(sites::AbstractVector) = (v = _sitevector(sites); InfiniteChain{eltype(v)}(v))
InfiniteChain(V) = InfiniteChain([V])
InfiniteChain(lat::InfiniteChain) = lat

# a concretely typed copy where the sites allow one
_sitevector(sites) = collect(map(identity, sites))

# every site a `Basis`, all of one symmetry
function _checksites(sites::Vector)
    isempty(sites) && return sites
    all(s -> s isa Basis, sites) || throw(
        ArgumentError(
            "a lattice is described by one operator `Basis` per site (a stock basis such as " *
                "`spin_basis(SU2Irrep)`, or `Basis(ops)`), got $(unique(typeof.(sites))); a bare " *
                "space no longer names an operator alphabet"
        )
    )
    I = sectortype(_space(first(sites)))
    all(s -> sectortype(_space(s)) === I, sites) || throw(
        ArgumentError(
            "all sites of a lattice must share one symmetry, got $(unique(sectortype.(_space.(sites))))"
        )
    )
    return sites
end

Base.length(lat::AbstractLattice) = length(lat.sites)
Base.isempty(lat::AbstractLattice) = isempty(lat.sites)
Base.eltype(::Type{FiniteChain{S}}) where {S} = S
Base.eltype(::Type{InfiniteChain{S}}) where {S} = S
Base.iterate(lat::AbstractLattice, args...) = iterate(lat.sites, args...)
Base.firstindex(::AbstractLattice) = 1
Base.lastindex(lat::AbstractLattice) = length(lat)
Base.:(==)(a::AbstractLattice, b::AbstractLattice) =
    typeof(a) == typeof(b) && a.sites == b.sites

Base.getindex(lat::FiniteChain, i::Integer) = lat.sites[i]
Base.getindex(lat::InfiniteChain, i::Integer) = lat.sites[mod1(Int(i), length(lat))]

TensorKit.sectortype(lat::AbstractLattice) = _lattice_sectortype(lat)
_lattice_sectortype(lat::AbstractLattice) =
    isempty(lat.sites) ? Trivial : sectortype(_space(first(lat.sites)))

Base.show(io::IO, lat::FiniteChain) = print(io, "FiniteChain(", lat.sites, ")")
Base.show(io::IO, lat::InfiniteChain) = print(io, "InfiniteChain(", lat.sites, ")")

"""
    windowlattice(lat::InfiniteChain, N::Int) -> FiniteChain

The first `N` sites of the infinite lattice as a finite chain (the unrolled window of the infinite sweep).
"""
windowlattice(lat::InfiniteChain, N::Int) = FiniteChain([lat[i] for i in 1:N])

# `nothing` for a lattice that has no edge; the site range a term must respect otherwise.
_maxsite(lat::FiniteChain) = length(lat)
_maxsite(::InfiniteChain) = nothing
