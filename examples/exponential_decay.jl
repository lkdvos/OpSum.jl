# # Exponentially decaying interactions
#
# A term sum holds couplings of finite range: every term names its sites, so an interaction reaching
# arbitrarily far needs infinitely many terms. There is exactly one infinite-range family an MPO
# represents *exactly* and at *fixed* cost — a geometric decay, because a constant on the diagonal of
# a bond channel is what an MPO multiplies by once per site:
#
# ```math
# \sum_{i<j} \lambda^{\,j-i-1}\, A_i\, S_{i+1}\cdots S_{j-1}\, B_j ,\qquad 0 < |\lambda| < 1 .
# ```
#
# [`expterm`](@ref OpSum.expterm) declares one such family. You write a single **representative
# term** and say where the stretched gap goes; the charges and the caterpillar fusion tree come from
# that term, exactly as for a finite-range coupling.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

V = SU2Space(1 // 2 => 1)
S = spin(V)

# ## One channel
#
# The representative is an ordinary two-site term. `decay` stretches the gap before its exit block.

tail = expterm(dot(S[1], S[2]); decay = 0.6)

# Adding it to a term bag gives a [`MixedSum`](@ref OpSum.MixedSum) — still latticeless, like
# everything else — which `irrep_mpo` accepts on either lattice.

H = dot(S[1], S[2]) + tail
irrep_mpo(H, InfiniteChain([V]))

# One extra bond index buys the entire tail. Compare the same model without it:

let cell = InfiniteChain([V])
    bare = irrep_mpo(dot(S[1], S[2]), cell)
    with = irrep_mpo(H, cell)
    println("  nearest neighbour only:  D=", map(length, bare.bondsectors))
    println("  + geometric tail:        D=", map(length, with.bondsectors))
end

# ## The same declaration on a finite chain
#
# The lattice fixes the translation period, and nothing else changes. On a
# [`FiniteChain`](@ref OpSum.FiniteChain) every site is a possible entry and the operator is the
# geometric sum **truncated to the chain**; on an `L`-site `InfiniteChain` entry and exit both step
# by `L`.

irrep_mpo(H, FiniteChain(V, 8))

# The finite MPO is not translation-invariant at the edges — the tail has fewer places to start near
# the right end — which is why its bond dimensions taper. The interior is what the infinite cell
# reproduces.
#
# Being a truncated geometric sum is a statement about which terms exist, and
# [`OpSum.chain_terms`](@ref) writes them out. So the finite case is checkable the ordinary way:

let lat = FiniteChain(V, 6)
    Ws, secs = irrep_mpo(H, lat)
    mpo_terms(Ws, secs) ≈ OpSum.chain_terms(H, length(lat))
end

# ## `decay` counts per site
#
# This is the one thing to get right. `λ` is applied once per *string* site, so the member with gap
# `g` (that is, `g` sites between the two operators) carries `λ^g`. Reading it off the reconstructed
# terms is the least ambiguous way to see it. Coefficients are quoted relative to the two-site term
# `dot` itself produces, so the Cartesian factor of the `dot` convention divides out:

base = only(dot(S[1], S[2])).coeff

let lat = FiniteChain(V, 5), λ = 0.5
    Ws, secs = irrep_mpo(expterm(dot(S[1], S[2]); decay = λ), lat)
    for t in mpo_terms(Ws, secs)
        gap = t.sites[end] - t.sites[1] - 1
        println("  sites=", t.sites, "  gap=", gap, "  coeff/base=", round(real(t.coeff / base); digits = 6))
    end
end

# Every entry site appears, and each extra site of separation costs one factor of `λ`.
#
# ## One `expterm` is a family; one `Terms` is a term
#
# Worth stating on its own, because the two halves of a `MixedSum` behave differently on a finite
# chain. `dot(S[1], S[2])` is a *single* term on sites 1 and 2 — a term bag is literal — while the
# `expterm` declares its whole family. So adding both does not give a uniform nearest-neighbour
# coupling; it doubles the coefficient at sites `[1, 2]` and nowhere else:

let lat = FiniteChain(V, 4), λ = 0.5
    Ws, secs = irrep_mpo(dot(S[1], S[2]) + expterm(dot(S[1], S[2]); decay = λ), lat)
    for t in mpo_terms(Ws, secs)
        println("  sites=", t.sites, "  coeff/base=", round(real(t.coeff / base); digits = 6))
    end
end

# To add a uniform nearest-neighbour coupling on top of a tail, write the nearest-neighbour terms out
# over the chain as usual — or, since the tail already contains a gap-0 member at every bond, simply
# scale the `expterm` and let it supply them.
#
# On an `InfiniteChain` the asymmetry disappears, because there the `Terms` part is a *generating
# set* and is translated too. Same declaration, different meaning of the operator — which is exactly
# what the lattice argument decides.

# ## A string operator in the gap
#
# By default the stretched sites carry the identity. `string` puts an on-site operator there instead
# — the one thing it must be is charge-neutral, since a charged string would make the running bond
# charge drift along the loop.

Vu = Rep[U₁](0 => 1, 1 => 1)
up, dn = U1Irrep(1), U1Irrep(0)
Su = spin_ops(Vu, up, dn)
parity = 2 * Su.Sz            # charge-neutral: a diagonal on-site operator

Hstring = couple(Su.Sp[1], Su.Sm[2]) +
    expterm(couple(Su.Sp[1], Su.Sm[2]); decay = 0.5, string = parity)
irrep_mpo(Hstring, InfiniteChain([Vu]))

# ## Multi-site blocks on either end
#
# `exitsite` is the first site of the exit block; everything before it is the entry block. So a
# three-site representative can stretch the gap between a two-site entry and a one-site exit:

Hblock = expterm(
    couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[4]); decay = 0.5, exitsite = 4
)
irrep_mpo(dot(S[1], S[2]) + Hblock, InfiniteChain([V]))

# ## A sum of exponentials
#
# Real power-law tails are often fitted by a handful of exponentials. Channels with different `λ` are
# linearly independent, so each costs one bond index; channels sharing `(λ, string, exit)` merge into
# one.

let cell = InfiniteChain([V]), λs = (0.9, 0.5, 0.1)
    h = dot(S[1], S[2])
    for λ in λs
        h = h + expterm(dot(S[1], S[2]); decay = λ)
    end
    println("  three decays:   D=", map(length, irrep_mpo(h, cell).bondsectors))
    dup = dot(S[1], S[2]) + expterm(dot(S[1], S[2]); decay = 0.5) +
        expterm(dot(S[1], S[2]); decay = 0.5)
    println("  two identical:  D=", map(length, irrep_mpo(dup, cell).bondsectors))
end

# So the cost is one index per *distinct* channel, and a fitted power law costs what its fit needs —
# which is the practical alternative to the exact all-to-all treatment on the
# [Long-range interactions](long_range.md) page, where the bond dimension grows with `N`.

# ## What is rejected
#
# `|λ| ≥ 1` does not converge, and is an error rather than a silently wrong operator:

try
    expterm(dot(S[1], S[2]); decay = 1.0)
catch e
    println(sprint(showerror, e))
end

# A charged string is rejected for the reason above:

try
    expterm(couple(Su.Sp[1], Su.Sm[2]); decay = 0.5, string = Su.Sp)
catch e
    println(first(sprint(showerror, e), 160))
end
