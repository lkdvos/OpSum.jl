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

B = spin_basis(SU2Irrep)
ex = spin_ops(SU2Irrep).S_exchange

# ## One channel
#
# The representative is an ordinary two-site term. `decay` stretches the gap before its exit block.

tail = expterm(ex[1, 2]; decay = 0.6)

# It goes into an [`OperatorSum`](@ref OpSum.OperatorSum) next to the finite-range terms, on either
# lattice — the channel is just another argument of `opsum`.

H = opsum(InfiniteChain(B), ex[1, 2], tail)
mpo(H)

# One extra bond index buys the entire tail. Compare the same model without it:

let cell = InfiniteChain(B)
    bare = mpo(opsum(cell, ex[1, 2]))
    with = mpo(H)
    println("  nearest neighbour only:  D=", map(length, bare.bondsectors))
    println("  + geometric tail:        D=", map(length, with.bondsectors))
end

# ## The same declaration on a finite chain
#
# The lattice fixes the translation period, and nothing else changes — only the first argument of
# `opsum`. On a
# [`FiniteChain`](@ref OpSum.FiniteChain) every site is a possible entry and the operator is the
# geometric sum **truncated to the chain**; on an `L`-site `InfiniteChain` entry and exit both step
# by `L`.

Hfinite = opsum(FiniteChain(B, 8), ex[1, 2], tail)
mpo(Hfinite)

# The finite MPO is not translation-invariant at the edges — the tail has fewer places to start near
# the right end — which is why its bond dimensions taper. The interior is what the infinite cell
# reproduces.
#
# Being a truncated geometric sum is a statement about which terms exist, and
# [`OpSum.chain_terms`](@ref) writes them out. So the finite case is checkable the ordinary way:

let H6 = opsum(FiniteChain(B, 6), ex[1, 2], tail)
    m = mpo(H6)
    mpo_terms(m) ≈ OpSum.chain_terms(H6)
end

# ## `decay` counts per site
#
# This is the one thing to get right. `λ` is applied once per *string* site, so the member with gap
# `g` (that is, `g` sites between the two operators) carries `λ^g`. Reading it off the reconstructed
# terms is the least ambiguous way to see it. Coefficients are quoted relative to the two-site term
# `ex[1, 2]` itself produces, so the normalisation of the exchange operator divides out:

base = only(ex[1, 2]).coeff

let lat = FiniteChain(B, 5), λ = 0.5
    m = mpo(opsum(lat, expterm(ex[1, 2]; decay = λ)))
    for t in mpo_terms(m)
        gap = t.sites[end] - t.sites[1] - 1
        println("  sites=", t.sites, "  gap=", gap, "  coeff/base=", round(real(t.coeff / base); digits = 6))
    end
end

# Every entry site appears, and each extra site of separation costs one factor of `λ`.
#
# ## One `expterm` is a family; one `Terms` is a term
#
# Worth stating on its own, because the two halves of an `OperatorSum` behave differently on a finite
# chain. `ex[1, 2]` is a *single* term on sites 1 and 2 — a term bag is literal — while the
# `expterm` declares its whole family. So adding both does not give a uniform nearest-neighbour
# coupling; it doubles the coefficient at sites `[1, 2]` and nowhere else:

let lat = FiniteChain(B, 4), λ = 0.5
    m = mpo(opsum(lat, ex[1, 2], expterm(ex[1, 2]; decay = λ)))
    for t in mpo_terms(m)
        println("  sites=", t.sites, "  coeff/base=", round(real(t.coeff / base); digits = 6))
    end
end

# To add a uniform nearest-neighbour coupling on top of a tail, write the nearest-neighbour terms out
# over the chain as usual — or, since the tail already contains a gap-0 member at every bond, simply
# scale the `expterm` and let it supply them.
#
# On an `InfiniteChain` the asymmetry disappears, because there the `Terms` part is a *generating
# set* and is translated too. Same declaration, different meaning of the operator — which is exactly
# what the lattice decides.

# ## A string operator in the gap
#
# By default the stretched sites carry the identity. `string` puts an on-site operator there instead
# — the one thing it must be is charge-neutral, since a charged string would make the running bond
# charge drift along the loop.

Bu = spin_basis(U1Irrep)
ou = spin_ops(U1Irrep)
parity = 2 * ou.S_z            # charge-neutral: a diagonal on-site operator

Hstring = opsum(
    InfiniteChain(Bu),
    ou.S_plus_S_min[1, 2],
    expterm(ou.S_plus_S_min[1, 2]; decay = 0.5, string = parity)
)
mpo(Hstring)

# ## Multi-site blocks on either end
#
# `exitsite` is the first site of the exit block; everything before it is the entry block. So a
# three-site representative can stretch the gap between a two-site entry and a one-site exit. Here
# the representative is the product ``S^z_1 S^z_2 \cdot S^z_4``, with the stretched gap at site 3:

Hblock = expterm(ou.S_z_S_z[1, 2] * ou.S_z[4]; decay = 0.5, exitsite = 4)
mpo(opsum(InfiniteChain(Bu), ou.S_z_S_z[1, 2], Hblock))

# ## A sum of exponentials
#
# Real power-law tails are often fitted by a handful of exponentials. Channels with different `λ` are
# linearly independent, so each costs one bond index; channels sharing `(λ, string, exit)` merge into
# one.

let cell = InfiniteChain(B), λs = (0.9, 0.5, 0.1)
    h = opsum(cell, ex[1, 2], (expterm(ex[1, 2]; decay = λ) for λ in λs))
    println("  three decays:   D=", map(length, mpo(h).bondsectors))
    dup = opsum(
        cell, ex[1, 2],
        expterm(ex[1, 2]; decay = 0.5), expterm(ex[1, 2]; decay = 0.5)
    )
    println("  two identical:  D=", map(length, mpo(dup).bondsectors))
end

# So the cost is one index per *distinct* channel, and a fitted power law costs what its fit needs —
# which is the practical alternative to the exact all-to-all treatment on the
# [Long-range interactions](long_range.md) page, where the bond dimension grows with `N`.

# ## What is rejected
#
# `|λ| ≥ 1` does not converge, and is an error rather than a silently wrong operator:

try
    expterm(ex[1, 2]; decay = 1.0)
catch e
    println(sprint(showerror, e))
end

# A charged string is rejected for the reason above. A charged operator is not something the
# builders hand out, but `project` accepts one written as a `TensorMap` with a charge leg:

Vu = space(Bu)
splus = zeros(ComplexF64, Vu ← Vu ⊗ Vect[U1Irrep](1 => 1))
for (_, blk) in blocks(splus)
    blk .= 1
end

try
    expterm(ou.S_plus_S_min[1, 2]; decay = 0.5, string = project(splus, Bu))
catch e
    println(first(sprint(showerror, e), 160))
end
