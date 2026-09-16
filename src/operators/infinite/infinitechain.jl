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
maxspan(ts::Terms) = maximum(termspan, ts.terms; init = 0)

"""
    unitcell_terms(ts, L::Int) -> Terms

Canonicalise a generating term bag for a unit cell of `L` sites: shift each term so that its
*leftmost* active site lies in `1:L`.

The result is a [`Terms`](@ref) bag whose leftmost
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

"""
    unitcell_terms(H::MixedSum, L::Int) -> MixedSum

Canonicalise a mixed generating set: the finite-range terms as above, and every exponentially decaying
channel shifted so that its *entry anchor* (the leftmost site of its representative) lies in `1:L`. The
same two rules apply to a channel as to a term — charge neutrality, and exactly one representative per
translation class, since `Σ_n translate(H, n·L)` would otherwise count a channel twice.
"""
function unitcell_terms(H::MixedSum{I}, L::Int) where {I}
    L ≥ 1 || throw(ArgumentError("unit cell length must be positive, got $L"))
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for (k, v) in pairs(H.channels.channels)
        total(k.term) == unit(I) || throw(
            ArgumentError(
                "infinite MPO construction requires charge-neutral terms, got total charge " *
                    "$(total(k.term)) on the channel with sites $(k.term.sites); charged infinite " *
                    "MPOs are out of scope"
            )
        )
        ck = translate(k, -L * fld(minimum(k.term.sites) - 1, L))
        haskey(d, ck) && throw(
            ArgumentError(
                "the exponentially decaying channels on sites $(k.term.sites) and " *
                    "$(ck.term.sites) are $L-translates of each other; the unit-cell generating set " *
                    "must contain exactly one representative per translation class (`H` means " *
                    "`Σ_n translate(H, n·$L)`)"
            )
        )
        insert!(d, ck, v)
    end
    return MixedSum{I}(unitcell_terms(H.terms, L), ExpSum{I}(d))
end

# a Hamiltonian of nothing but channels behaves like a `MixedSum` with no terms
unitcell_terms(H::ExpSum, L::Int) = unitcell_terms(MixedSum(H), L)
window_terms(H::ExpSum, lat::InfiniteChain, nc::Int) = window_terms(MixedSum(H), lat, nc)
maxspan(H::ExpSum) = maxspan(MixedSum(H))

"""
    maxspan(H::MixedSum) -> Int

Range of the *shortest* translates of a mixed generating set: the largest [`termspan`](@ref) over its
finite terms and [`channelspan`](@ref) over its channels. A channel's actual range is unbounded — this
is the range over which the sweep has to settle into its periodic fixed point.
"""
maxspan(H::MixedSum) = max(
    maxspan(H.terms), maximum(channelspan, keys(H.channels.channels); init = 0)
)

"""
    window_terms(H::MixedSum, lat::InfiniteChain, ncells::Int) -> Terms

Every finite-range translate that fits inside the window, plus every translate of every channel that
fits (see [`expand_channels`](@ref)) — the explicit expansion the faithfulness check compares against.
"""
function window_terms(H::MixedSum{I}, lat::InfiniteChain, ncells::Int) where {I}
    N = ncells * length(lat)
    return opsum(
        _window_bag(H.terms, length(lat), N),
        expand_channels(H.channels, length(lat), N),
    )
end

"""
    window_terms(gen::Terms, lat::InfiniteChain, ncells::Int) -> Terms

Every `L`-translate of the canonical generating set `gen` whose support lies entirely inside the
window `1:(ncells*L)`. Translates that would stick out of either end
are dropped, so the result is *not* the infinite Hamiltonian truncated to the window — only its bulk
agrees with the periodic problem, which is exactly what the window construction reads off.
"""
function window_terms(gen::Terms{I}, lat::InfiniteChain, ncells::Int) where {I}
    N = ncells * length(lat)
    return _window_bag(gen, length(lat), N)
end

# The translates themselves, as an unbound bag — so a mixed generating set can concatenate them with
# its expanded channels and bind the lattice once.
function _window_bag(gen::Terms{I}, L::Int, N::Int) where {I}
    out = Term{I}[]
    for t in gen.terms
        lo, hi = minimum(t.sites), maximum(t.sites)
        for n in cld(1 - lo, L):fld(N - hi, L)
            push!(out, translate(t, n * L))
        end
    end
    return Terms{I}(out)
end
