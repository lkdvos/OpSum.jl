# Infinite chains with a repeating unit cell: translation, and the canonicalisation that turns a
# user-written term sum into a generating set with one representative per translation class.
#
#     mpo(opsum(InfiniteChain(bases), H))   represents   Σ_{n ∈ ℤ} translate(H, n·L),   L = length(spaces)
#
# so `H` is a *generating set*, not the Hamiltonian: writing both `dot(S[1], S[2])` and
# `dot(S[2], S[3])` on a one-site cell would double-count, and is rejected.

using TensorKit: Sector, ElementarySpace, unit

"Shift a term by `Δ` sites. Only the sites change, so the fusion data is undisturbed."
translate(t::Term{I}, Δ) where {I} = Term{I}(t.sites .+ Int(Δ), t.keys, t.coeff)

"Shift every term of a bag. The result is latticeless: a translate generally leaves its operand's window."
translate(ts::Terms{I}, Δ) where {I} =
    iszero(Δ) ? ts : Terms{I}([translate(t, Δ) for t in ts.terms])

"Number of bonds between a term's first and last active site (`0` if it has none)."
termspan(t::Term) = isempty(t.sites) ? 0 : maximum(t.sites) - minimum(t.sites)

"The interaction range of a term bag: the largest [`termspan`](@ref) over its terms."
maxspan(ts::Terms) = maximum(termspan, ts.terms; init = 0)

# Terms must be charge-neutral: the sweep's running bond charge is referenced to the left vacuum, which
# only for a neutral term agrees with "charge accumulated since the class entered".
function _checkneutral(t::Term{I}) where {I}
    return total(t) == unit(I) || throw(
        ArgumentError(
            "infinite MPO construction requires charge-neutral terms, got total charge " *
                "$(total(t)) on sites $(t.sites); charged infinite MPOs are out of scope"
        )
    )
end

# What a generating term must satisfy on its own, checked on insertion and again when the cell is formed.
function _checkgenerator(t::Term)
    isempty(t.sites) && throw(
        ArgumentError(
            "an infinite MPO cannot represent a K = 0 identity term: `Σ_n c·𝟙` does not converge"
        )
    )
    return _checkneutral(t)
end
_checkgenerator(k::ExpKey) = _checkneutral(k.term)

# `-L * fld(first site - 1, L)`: the shift putting the leftmost site into `1:L`.
_cellshift(sites, L) = -L * fld(minimum(sites) - 1, L)

"""
    unitcell_terms(ts, L::Int) -> Terms

Shift each term of a generating bag so that its leftmost active site lies in `1:L` (the rightmost may
reach `L + R`, so the result is latticeless). Two terms that are `L`-translates of each other would be
counted twice in `Σ_n translate(H, n·L)`, so that is an error. Terms must be charge-neutral and not
`K = 0`.
"""
function unitcell_terms(ts::Terms{I}, L::Int) where {I}
    L ≥ 1 || throw(ArgumentError("unit cell length must be positive, got $L"))
    seen = Dictionary{Term{I}, Int}()
    out = Term{I}[]
    # canonical, so that only genuine translates of each other collide below
    for t in canonicalize!(ts).terms
        _checkgenerator(t)
        ct = translate(t, _cellshift(t.sites, L))
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
    unitcell_terms(es::ExpSum, L::Int) -> ExpSum

As for terms: every channel is shifted so that its entry anchor lies in `1:L`; the same neutrality and
one-representative-per-class rules apply.
"""
function unitcell_terms(es::ExpSum{I}, L::Int) where {I}
    L ≥ 1 || throw(ArgumentError("unit cell length must be positive, got $L"))
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for (k, v) in pairs(es.channels)
        _checkgenerator(k)
        ck = translate(k, _cellshift(k.term.sites, L))
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
    return ExpSum{I}(d)
end

"""
    unitcell_terms(H::OperatorSum{I, <:InfiniteChain}) -> OperatorSum

`H` in unit-cell form, on the same lattice. The one-representative-per-class rule is enforced here.
"""
function unitcell_terms(H::OperatorSum{I, <:InfiniteChain}) where {I}
    L = length(H.lattice)
    return OperatorSum{I, typeof(H.lattice)}(
        H.lattice, unitcell_terms(H.terms, L), unitcell_terms(H.channels, L)
    )
end

maxspan(es::ExpSum) = maximum(channelspan, keys(es.channels); init = 0)

"""
    maxspan(H::OperatorSum) -> Int

Range of the *shortest* translates of a generating set (terms and channels): the range over which the
sweep has to settle into its periodic fixed point. A channel's actual range is unbounded.
"""
maxspan(H::OperatorSum) = max(maxspan(H.terms), maxspan(H.channels))

"""
    window_terms(H::OperatorSum{I, <:InfiniteChain}, ncells::Int) -> Terms

Every translate of the generating set (terms and channels, see [`expand_channels`](@ref)) that fits
inside a window of `ncells` unit cells: the explicit expansion the faithfulness check compares against.
"""
function window_terms(H::OperatorSum{I, <:InfiniteChain}, ncells::Int) where {I}
    L = length(H.lattice)
    N = ncells * L
    return opsum(_window_bag(H.terms, L, N), expand_channels(H.channels, L, N))
end

"""
    window_terms(gen::Terms, lat::InfiniteChain, ncells::Int) -> Terms

Every `L`-translate of the canonical generating set `gen` whose support lies inside `1:(ncells*L)`.
Translates sticking out of either end are dropped, so only the bulk agrees with the periodic problem.
"""
window_terms(gen::Terms, lat::InfiniteChain, ncells::Int) =
    _window_bag(gen, length(lat), ncells * length(lat))

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
