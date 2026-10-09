# The lattices an operator can be compressed over. A term bag is latticeless — nothing in the term
# algebra or in the sweep needs a physical space, and the sweep needs only the site *count* — so the
# lattice enters with the `OperatorSum`, and these are the two things that can be supplied.
#
# A site is described by its operator `Basis` (basis.jl), whose letters the terms on it index into, or
# — until the matrix-unit alphabet goes (operator-bases.md §7, step 7) — by its bare physical space,
# whose matrix units the letters index into. A lattice is one or the other throughout.
#
# `FiniteChain` and `InfiniteChain` differ in exactly two observable ways: whether `getindex` wraps,
# and whether a site index outside `1:length` is an error. Everything downstream branches on the type
# rather than on a flag.

using TensorKit: TensorKit, ElementarySpace, Sector, Trivial, sectortype

"""
    AbstractLattice

Supertype of the lattices an [`OperatorSum`](@ref) can carry: [`FiniteChain`](@ref) and
[`InfiniteChain`](@ref). A lattice names the operator [`Basis`](@ref) of every site (or, for the
matrix-unit alphabet, its physical space), which is what tensor assembly, [`instantiate`](@ref) and
letter validation need; the compression itself needs only `length` (the number of sites, or the
unit-cell period).
"""
abstract type AbstractLattice end

"""
    FiniteChain(bases::AbstractVector{<:Basis})
    FiniteChain(B::Basis, N::Integer)
    FiniteChain(spaces::AbstractVector{<:ElementarySpace})
    FiniteChain(V::ElementarySpace, N::Integer)

An open chain of `length(bases)` sites, carrying one operator [`Basis`](@ref) per site: the letters
of a term on site `i` index into `bases[i]`. Sites with different spaces are combined by listing
their bases, e.g. `vcat(Bf, Bs, Bf)`; all must share one symmetry. `getindex` does *not* wrap: site
`i` exists only for `i ∈ 1:length(chain)`, and a term reaching outside that range is an error rather
than a wrap-around.

With spaces instead of bases the letters are matrix units of those spaces (to be removed with the
matrix-unit alphabet). `FiniteChain(B, N)` is the uniform chain. A bare vector of bases or spaces is
accepted anywhere a `FiniteChain` is, and converted.
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

An infinite chain with a repeating unit cell of `L = length(bases)` sites, carrying one operator
[`Basis`](@ref) per site of the cell (or, for the matrix-unit alphabet, one space). Site `i` of the
infinite lattice has basis `bases[mod1(i, L)]`, for any `i ∈ ℤ` — `getindex` wraps, so this is the
lattice of the whole chain and not just of the cell.

An operator compressed over one is a *generating set*: the operator represented is
`Σ_{n ∈ ℤ} translate(H, n·L)`, so each translation class must appear exactly once (see
[`unitcell_terms`](@ref)). Its terms may reach past the cell — `dot(S[1], S[2])` on a one-site cell
names site 2 — which is exactly what the wrap-around `getindex` is for.
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

# A concretely typed copy where the sites allow one (`vcat` of bases of different spaces need not).
_sitevector(sites) = collect(map(identity, sites))

# One kind of site throughout, of one symmetry. `_space`/`Basis` are defined later (basis.jl), which
# is fine for a function only called at run time.
function _checksites(sites::Vector)
    isempty(sites) && return sites
    bases = all(s -> s isa Basis, sites)
    bases || all(s -> s isa ElementarySpace, sites) || throw(
        ArgumentError(
            "a lattice is described by one operator `Basis` per site or (for the matrix-unit " *
                "alphabet) one `ElementarySpace` per site, not a mix; got $(unique(typeof.(sites)))"
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

# The one behavioural difference: the finite chain has an edge, the infinite one does not.
Base.getindex(lat::FiniteChain, i::Integer) = lat.sites[i]
Base.getindex(lat::InfiniteChain, i::Integer) = lat.sites[mod1(Int(i), length(lat))]

TensorKit.sectortype(lat::AbstractLattice) = _lattice_sectortype(lat)
_lattice_sectortype(lat::AbstractLattice) =
    isempty(lat.sites) ? Trivial : sectortype(_space(first(lat.sites)))

Base.show(io::IO, lat::FiniteChain) = print(io, "FiniteChain(", lat.sites, ")")
Base.show(io::IO, lat::InfiniteChain) = print(io, "InfiniteChain(", lat.sites, ")")

"""
    windowlattice(lat::InfiniteChain, N::Int) -> FiniteChain

The first `N` sites of the infinite lattice as a finite chain, for the unrolled window the infinite
sweep runs on.
"""
windowlattice(lat::InfiniteChain, N::Int) = FiniteChain([lat[i] for i in 1:N])

# `nothing` for a lattice that has no edge; the site range a term must respect otherwise.
_maxsite(lat::FiniteChain) = length(lat)
_maxsite(::InfiniteChain) = nothing
