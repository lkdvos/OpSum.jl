# Infinite chains with a repeating unit cell: the lattice descriptor and the term-level operations an
# infinite MPO needs on top of the finite pipeline. Nothing here touches the sweep — a term over `Int`
# site labels already ranges over all of ℤ, so the only new operations are *translation* and the
# canonicalisation that turns a user-written term sum into a generating set with one representative
# per translation class.
#
#     irrep_mpo(H, InfiniteChain(spaces))   represents   Σ_{n ∈ ℤ} translate(H, n·L),   L = length(spaces)
#
# so `H` is a *generating set*, not the Hamiltonian: writing both `dot(S[1], S[2])` and
# `dot(S[2], S[3])` on a one-site cell would double-count, and is rejected.

using TensorKit: Sector, ElementarySpace, unit

"""
    InfiniteChain(spaces::AbstractVector)

An infinite chain with a repeating unit cell of `L = length(spaces)` sites, carrying one physical
space per site of the cell. Site `i` of the infinite lattice has space `spaces[mod1(i, L)]`, for any
`i ∈ ℤ` — `getindex` wraps, so it is the lattice of the whole chain and not just of the cell.

This is *not* the lattice a generating `TermSum` carries. That one only has to be long enough to host
the generating terms, which reach up to `R` sites past the cell (`dot(S[1], S[2])` on a one-site cell
needs two spaces); the cell length `L` comes from here.
"""
struct InfiniteChain{S <: ElementarySpace}
    spaces::Vector{S}
    function InfiniteChain(spaces::AbstractVector{S}) where {S <: ElementarySpace}
        isempty(spaces) && throw(ArgumentError("an InfiniteChain needs at least one site per unit cell"))
        return new{S}(collect(spaces))
    end
end
InfiniteChain(V::ElementarySpace) = InfiniteChain([V])

Base.length(lat::InfiniteChain) = length(lat.spaces)
Base.getindex(lat::InfiniteChain, i::Integer) = lat.spaces[mod1(Int(i), length(lat))]
Base.eltype(::Type{InfiniteChain{S}}) where {S} = S
Base.iterate(lat::InfiniteChain, args...) = iterate(lat.spaces, args...)
sectortype(lat::InfiniteChain) = sectortype(eltype(lat.spaces))

Base.show(io::IO, lat::InfiniteChain) = print(io, "InfiniteChain(", lat.spaces, ")")

"""
    windowlattice(lat::InfiniteChain, N::Int) -> Vector{<:ElementarySpace}

The first `N` sites of the infinite lattice as a finite lattice, for binding an unrolled window with
[`opsum`](@ref).
"""
windowlattice(lat::InfiniteChain, N::Int) = [lat[i] for i in 1:N]

# `opsum` checked every letter against the space of the site it sits on — but against *H's* lattice,
# which is only as long as the generating terms reach. The cell is declared separately, so the two can
# disagree; that would compress the operator on the wrong spaces, silently.
function _check_chain_lattice(H::TermSum, chain::InfiniteChain)
    lat = lattice(H)
    for i in eachindex(lat)
        lat[i] == chain[i] || throw(
            ArgumentError(
                "site $i carries space $(lat[i]) in the operator's lattice but $(chain[i]) in the " *
                    "unit cell (site $(mod1(i, length(chain))) of $(length(chain))); the generating " *
                    "set must be written on the same chain it is tiled over"
            )
        )
    end
    return nothing
end

"""
    translate(t::Term, Δ) -> Term

Shift a term by `Δ` sites. Only the sites change: the letters, their running bond charges and the
coefficient carry over untouched, so translation cannot disturb the fusion data and is `Θ(K)`.
"""
translate(t::Term{I}, Δ) where {I} = Term{I}(t.sites .+ Int(Δ), t.keys, t.coeff)

"""
    translate(ts::Terms, Δ) -> Terms

Shift every term of a bag. The result is latticeless by construction: a translate generally leaves the
window its operand was bound to.
"""
translate(ts::Terms{I}, Δ) where {I} =
    iszero(Δ) ? ts : Terms{I}([translate(t, Δ) for t in ts.terms])
translate(H::TermSum{I}, Δ) where {I} =
    iszero(Δ) ? Terms{I}(copy(H.terms)) : Terms{I}([translate(t, Δ) for t in H.terms])

"""
    termspan(t::Term) -> Int

Number of bonds between a term's first and last active site (`0` for a `K ≤ 1` term).
"""
termspan(t::Term) = isempty(t.sites) ? 0 : maximum(t.sites) - minimum(t.sites)

"""
    maxspan(ts) -> Int

The interaction range `R` of a term bag or sum: the largest [`termspan`](@ref) over its terms. This is
what sets how much padding an unrolled window needs, and how many unit cells the sweep can take to
reach its fixed point.
"""
maxspan(ts::Union{Terms, TermSum}) = maximum(termspan, ts.terms; init = 0)

"""
    unitcell_terms(ts, L::Int) -> Terms

Canonicalise a generating term bag for a unit cell of `L` sites: shift each term so that its
*leftmost* active site lies in `1:L`.

The result is a latticeless [`Terms`](@ref) bag rather than a `TermSum`: a canonical term's leftmost
site is in `1:L` but its rightmost may reach `L + R`, so there is no `L`-site lattice to bind it to.
Its input is latticeless for the same reason — on an infinite chain the [`InfiniteChain`](@ref) names
the space of every site, so a generating set carries no lattice of its own.

Two terms of `ts` that are `L`-translates of each other collapse onto the same canonical term, which
means `Σ_n translate(H, n·L)` counts that translation class twice. That is almost always a mistake
(e.g. writing both `dot(S[1], S[2])` and `dot(S[2], S[3])` on a one-site cell), so it is an error
rather than a silent doubling.

Terms are also required to be charge-neutral (`total == unit(I)`): the running bond charge carried by
the sweep is referenced to the left vacuum, and only for a neutral Hamiltonian does that agree with
"charge accumulated since the class entered", which is the translation-invariant notion. `K = 0`
identity terms are rejected too — on an infinite lattice `Σ_n c·𝟙` does not converge.
"""
function unitcell_terms(ts::Terms{I}, L::Int) where {I}
    L ≥ 1 || throw(ArgumentError("unit cell length must be positive, got $L"))
    seen = Dictionary{Term{I}, Int}()
    out = Term{I}[]
    for t in ts.terms
        isempty(t.sites) && throw(
            ArgumentError(
                "an infinite MPO cannot represent a K = 0 identity term: `Σ_n c·𝟙` does not converge"
            )
        )
        total(t) == unit(I) || throw(
            ArgumentError(
                "infinite MPO construction requires charge-neutral terms, got total charge " *
                    "$(total(t)) on sites $(t.sites); charged infinite MPOs are out of scope"
            )
        )
        ct = translate(t, -L * fld(minimum(t.sites) - 1, L))
        prev = get(seen, ct, 0)
        iszero(prev) || throw(
            ArgumentError(
                "terms on sites $(t.sites) and $(out[prev].sites) are $L-translates of each other; " *
                    "the unit-cell generating set must contain exactly one representative per " *
                    "translation class (`H` means `Σ_n translate(H, n·$L)`)"
            )
        )
        push!(out, ct)
        insert!(seen, ct, length(out))
    end
    return Terms{I}(out)
end

# A lattice-bound operator is accepted too, as long as its lattice agrees with the chain: `opsum`
# already checked its letters against it, so disagreeing spaces would mean the two checks were run
# against different chains.
function unitcell_terms(H::TermSum{I}, L::Int) where {I}
    return unitcell_terms(Terms{I}(H.terms), L)
end

"""
    window_terms(gen::Terms, lat::InfiniteChain, ncells::Int) -> TermSum

Every `L`-translate of the canonical generating set `gen` whose support lies entirely inside the
window `1:(ncells*L)`, bound to that window's lattice. Translates that would stick out of either end
are dropped, so the result is *not* the infinite Hamiltonian truncated to the window — only its bulk
agrees with the periodic problem, which is exactly what the window construction reads off.
"""
function window_terms(gen::Terms{I}, lat::InfiniteChain, ncells::Int) where {I}
    L = length(lat)
    N = ncells * L
    out = Term{I}[]
    for t in gen.terms
        lo, hi = minimum(t.sites), maximum(t.sites)
        for n in cld(1 - lo, L):fld(N - hi, L)
            push!(out, translate(t, n * L))
        end
    end
    return opsum(windowlattice(lat, N), Terms{I}(out))
end
