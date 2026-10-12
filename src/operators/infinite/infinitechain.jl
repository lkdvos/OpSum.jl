# Infinite chains: translation, and the canonicalisation of a generating set (one representative per
# translation class) for `Σ_n translate(H, n·L)`.

using TensorKit: Sector, ElementarySpace, unit

translate(t::Term{I}, Δ) where {I} = Term{I}(t.sites .+ Int(Δ), t.keys, t.coeff)
translate(ts::Terms{I}, Δ) where {I} =
    iszero(Δ) ? ts : Terms{I}([translate(t, Δ) for t in ts.terms])

termspan(t::Term) = isempty(t.sites) ? 0 : maximum(t.sites) - minimum(t.sites)
maxspan(ts::Terms) = maximum(termspan, ts.terms; init = 0)
maxspan(es::ExpSum) = maximum(channelspan, keys(es.channels); init = 0)
# range of the shortest translates (a channel's actual range is unbounded)
maxspan(H::OperatorSum) = max(maxspan(H.terms), maxspan(H.channels))

# The sweep's running bond charge is referenced to the left vacuum, so terms must be neutral.
function _checkneutral(t::Term{I}) where {I}
    return total(t) == unit(I) || throw(
        ArgumentError(
            "infinite MPO construction requires charge-neutral terms, got total charge " *
                "$(total(t)) on sites $(t.sites); charged infinite MPOs are out of scope"
        )
    )
end

function _checkgenerator(t::Term)
    isempty(t.sites) && throw(
        ArgumentError(
            "an infinite MPO cannot represent a K = 0 identity term: `Σ_n c·𝟙` does not converge"
        )
    )
    return _checkneutral(t)
end
_checkgenerator(k::ExpKey) = _checkneutral(k.term)

# shift putting the leftmost site into `1:L`
_cellshift(sites, L) = -L * fld(minimum(sites) - 1, L)

_translates_error(what, s1, s2, L) = throw(
    ArgumentError(
        "$what on sites $s1 and $s2 are $L-translates of each other; the unit-cell generating set " *
            "must contain exactly one representative per translation class " *
            "(`H` means `Σ_n translate(H, n·$L)`)"
    )
)

# Shift each generator so its leftmost site lies in `1:L`; two `L`-translates of each other are an error.
function unitcell_terms(ts::Terms{I}, L::Int) where {I}
    L ≥ 1 || throw(ArgumentError("unit cell length must be positive, got $L"))
    seen = Dictionary{Term{I}, Int}()
    out = Term{I}[]
    for t in canonicalize!(ts).terms
        _checkgenerator(t)
        ct = translate(t, _cellshift(t.sites, L))
        prev = get(seen, ct, 0)
        iszero(prev) || _translates_error("terms", t.sites, out[prev].sites, L)
        push!(out, ct)
        insert!(seen, ct, length(out))
    end
    return Terms{I}(out)
end

function unitcell_terms(es::ExpSum{I}, L::Int) where {I}
    L ≥ 1 || throw(ArgumentError("unit cell length must be positive, got $L"))
    d = Dictionary{ExpKey{I}, ComplexF64}()
    for (k, v) in pairs(es.channels)
        _checkgenerator(k)
        ck = translate(k, _cellshift(k.term.sites, L))
        haskey(d, ck) && _translates_error("the decaying channels", k.term.sites, ck.term.sites, L)
        insert!(d, ck, v)
    end
    return ExpSum{I}(d)
end

function unitcell_terms(H::OperatorSum{I, <:InfiniteChain}) where {I}
    L = length(H.lattice)
    return OperatorSum{I, typeof(H.lattice)}(
        H.lattice, unitcell_terms(H.terms, L), unitcell_terms(H.channels, L)
    )
end

# Every translate of the generating set (terms and channels) inside `ncells` cells.
function window_terms(H::OperatorSum{I, <:InfiniteChain}, ncells::Int) where {I}
    L = length(H.lattice)
    return opsum(window_terms(H.terms, H.lattice, ncells), expand_channels(H.channels, L, ncells * L))
end

# Translates sticking out of either end are dropped, so only the bulk agrees with the periodic problem.
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
    return Terms{I}(out)
end
