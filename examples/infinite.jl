# # Infinite chains
#
# Everything so far has been a finite chain. Swap the lattice for an `InfiniteChain` and the *same*
# term algebra builds an MPO with a repeating unit cell. Nothing about writing the operator changes
# — which is the point of the lattice being the first argument of `opsum` rather than something a
# term bag carries: the terms are the same bag, and `irrep_mpo(H)` has one signature for both.
#
# One thing does change, and it is a change of meaning rather than of spelling: the operator you
# write down is a **generating set**, not the Hamiltonian. What is represented is
#
# ```math
# \sum_{n \in \mathbb{Z}} \mathrm{translate}(H,\, nL)
# ```
#
# so each translation class must appear **exactly once**.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

using MatrixAlgebraKit: truncrank

V = SU2Space(1 // 2 => 1)
S = spin(V)

# ## A one-site cell
#
# The Heisenberg chain has one bond per site, so one representative on a one-site cell is the whole
# model. The call is shaped exactly like the finite one.

cell1 = InfiniteChain([V])
H∞ = irrep_mpo(opsum(cell1, dot(S[1], S[2])))

# What comes back is an [`InfiniteMPO`](@ref OpSum.InfiniteMPO) rather than a
# [`FiniteMPO`](@ref OpSum.FiniteMPO); it destructures as the same pair, with bond `0` identified
# with bond `L`.

Ws, secs = H∞
(; L = length(Ws), D = map(length, secs), Ddense = map(s -> sum(dim, s), secs))

# `D = 5` dense, `3` reduced — the same numbers as the *bulk* of the finite chain. That is the
# content of the fixed point: the sweep unrolls a window, runs the unchanged finite sweep, and finds
# the first cell that has become translation-invariant.

let fin = build("Heisenberg, finite N=16", opsum(FiniteChain(V, 16), dot(S[i], S[i + 1]) for i in 1:15); quiet = true)
    println("  finite bulk:  D=$(fin.D)  D_dense=$(fin.Ddense)")
    println("  infinite:     D=$(only(map(length, secs)))  D_dense=$(only(map(s -> sum(dim, s), secs)))")
end

# ## Double-counting is rejected, not silently doubled
#
# This is the one mistake the generating-set semantics invites, so it is an error rather than a
# factor of two in your Hamiltonian. On a one-site cell, `dot(S[1], S[2])` and `dot(S[2], S[3])` are
# the *same* translation class. Adding them is fine — each is a perfectly good term — it is forming
# the MPO that sees the pair together:

Hdouble = opsum(cell1, dot(S[1], S[2]), dot(S[2], S[3]))
try
    irrep_mpo(Hdouble)
catch e
    println(sprint(showerror, e))
end

# Charge neutrality is required too — a charged generator would make the running bond charge drift
# from cell to cell, so there is no fixed point to find. That one needs no global view, so it is
# refused the moment the term is added.

# ## A two-site cell
#
# On a two-site cell those same two bonds are genuinely different, and both are needed. That is what
# a bigger cell buys: couplings that alternate.

cell2 = InfiniteChain([V, V])
Hdimer = irrep_mpo(opsum(cell2, 0.6 * dot(S[1], S[2]), 1.4 * dot(S[2], S[3])))

# ```math
# H = \sum_{n} \left( J_1\, \vec{S}_{2n-1}\!\cdot\!\vec{S}_{2n} + J_2\, \vec{S}_{2n}\!\cdot\!\vec{S}_{2n+1} \right)
# ```

map(length, Hdimer.bondsectors)

# The dimerised chain costs the same bond dimension per bond as the uniform one; the cell is bigger,
# so there are two bond matrices instead of one.

# ## A non-uniform cell
#
# The cell is a list of spaces, so the sites within it need not be the same — an alternating
# spin-½ / spin-1 chain is a two-site cell with two different spaces.

V1 = SU2Space(1 => 1)
Sa, Sb = spin(V), spin(V1)
Halt = irrep_mpo(opsum(InfiniteChain([V, V1]), dot(Sa[1], Sb[2]), dot(Sb[2], Sa[3])))
map(length, Halt.bondsectors)

# ## Longer range
#
# Finite range is all that is required, not nearest neighbour. A ``J_1``–``J_2`` chain needs two
# generators on a one-site cell, one per translation class:

Hj1j2 = irrep_mpo(opsum(cell1, dot(S[1], S[2]), 0.5 * dot(S[1], S[3])))
map(length, Hj1j2.bondsectors)

# The window the sweep unrolls grows with the interaction range, but the returned cell does not: it
# is always `L` bond matrices.

# ## The boundary channels
#
# An infinite MPO carries the two identity-channel indices, which is what a consumer needs to start
# and terminate the chain: `start[j]` is the channel on which nothing has begun and `done[j]` the one
# on which everything has finished.

(; start = H∞.start, done = H∞.done)

# ## Assembling the tensors
#
# [`irrep_mpo_tensors`](@ref OpSum.irrep_mpo_tensors) gives `L` site tensors that **tile**: site 1's
# left virtual space is site `L`'s right virtual space. That property is exactly what an infinite
# MPS algorithm requires of its operator.

T∞ = irrep_mpo_tensors(H∞, cell1)
space(T∞[1], 1) == space(T∞[end], 4)'

# What you can do with them downstream — and the one thing you cannot yet — is on the
# [Handing the MPO to MPSKit](mpskit.md) page.

# ## What is not available here
#
# `SVDBondAlgorithm` is finite-only. It compresses each bond independently against a
# vacuum-terminated layout, and there is no bond basis that closes on itself, so there is nothing for
# the fixed point to converge to:

try
    irrep_mpo(opsum(cell1, dot(S[1], S[2])), SVDBondAlgorithm(truncrank(2)))
catch e
    println(first(sprint(showerror, e), 160))
end

# Terms with no support (`K = 0`) are rejected too, on insertion: ``\sum_n c\,\mathbb{1}`` does not
# converge.
#
# Infinite-range couplings are a different matter — a *geometric* decay is representable exactly, at
# fixed cost, and has its own page: [Exponentially decaying interactions](exponential_decay.md).
