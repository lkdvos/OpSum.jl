# # Infinite chains
#
# Everything so far has been a finite chain. Swap the lattice for an `InfiniteChain` and the *same*
# term algebra builds an MPO with a repeating unit cell. Nothing about writing the operator changes
# — which is the point of the lattice being the first argument of `opsum` rather than something a
# term bag carries: the terms are the same bag, and `mpo(H)` has one signature for both.
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

B = spin_basis(SU2Irrep)
ex = spin_ops(SU2Irrep).S_exchange

# ## A one-site cell
#
# The Heisenberg chain has one bond per site, so one representative on a one-site cell is the whole
# model. The call is shaped exactly like the finite one.

cell1 = InfiniteChain(B)
H∞ = mpo(opsum(cell1, ex[1, 2]))

# What comes back is a [`SymbolicMPO`](@ref OpSum.SymbolicMPO) on the infinite chain: the same bond
# matrices and bond charges as for a finite chain, for the `L` sites of the cell, with bond `0`
# identified with bond `L`.

Ws, secs = H∞.Ws, H∞.bondsectors
(; L = length(Ws), D = map(length, secs), Ddense = map(s -> sum(dim, s), secs))

# `D = 5` dense, `3` reduced — the same numbers as the *bulk* of the finite chain. That is the
# content of the fixed point: the sweep unrolls a window, runs the unchanged finite sweep, and finds
# the first cell that has become translation-invariant.

let fin = build("Heisenberg, finite N=16", opsum(FiniteChain(B, 16), ex[i, i + 1] for i in 1:15); quiet = true)
    println("  finite bulk:  D=$(fin.D)  D_dense=$(fin.Ddense)")
    println("  infinite:     D=$(only(map(length, secs)))  D_dense=$(only(map(s -> sum(dim, s), secs)))")
end

# ## Double-counting is rejected, not silently doubled
#
# This is the one mistake the generating-set semantics invites, so it is an error rather than a
# factor of two in your Hamiltonian. On a one-site cell, `ex[1, 2]` and `ex[2, 3]` are
# the *same* translation class. Adding them is fine — each is a perfectly good term — it is forming
# the MPO that sees the pair together:

Hdouble = opsum(cell1, ex[1, 2], ex[2, 3])
try
    mpo(Hdouble)
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

cell2 = InfiniteChain([B, B])
Hdimer = mpo(opsum(cell2, 0.6 * ex[1, 2], 1.4 * ex[2, 3]))

# ```math
# H = \sum_{n} \left( J_1\, \vec{S}_{2n-1}\!\cdot\!\vec{S}_{2n} + J_2\, \vec{S}_{2n}\!\cdot\!\vec{S}_{2n+1} \right)
# ```

map(length, Hdimer.bondsectors)

# The dimerised chain costs the same bond dimension per bond as the uniform one; the cell is bigger,
# so there are two bond matrices instead of one.

# ## A non-uniform cell
#
# The cell is a list of bases, so the sites within it need not be the same — an alternating
# spin-½ / spin-1 chain is a two-site cell with two different bases (the mixed bond `exchange` is
# the one from [Lattices and geometry](lattices.md)).

B1 = spin_basis(SU2Irrep; spin = 1)
Halt = mpo(opsum(InfiniteChain([B, B1]), exchange(B, B1)[1, 2], exchange(B1, B)[2, 3]))
map(length, Halt.bondsectors)

# ## Longer range
#
# Finite range is all that is required, not nearest neighbour. A ``J_1``–``J_2`` chain needs two
# generators on a one-site cell, one per translation class:

Hj1j2 = mpo(opsum(cell1, ex[1, 2], 0.5 * ex[1, 3]))
map(length, Hj1j2.bondsectors)

# The window the sweep unrolls grows with the interaction range, but the returned cell does not: it
# is always `L` bond matrices.

# ## The boundary channels
#
# An infinite MPO carries the two identity-channel indices, which is what a consumer needs to start
# and terminate the chain: `start[j]` is the channel on which nothing has begun and `finish[j]` the
# one on which everything has finished.

(; start = H∞.start, finish = H∞.finish)

# ## Assembling the tensors
#
# [`mpo_tensors`](@ref OpSum.mpo_tensors) gives `L` site tensors that **tile**: site 1's
# left virtual space is site `L`'s right virtual space. That property is exactly what an infinite
# MPS algorithm requires of its operator.

T∞ = mpo_tensors(H∞)
space(T∞[1], 1) == space(T∞[end], 4)'

# What you can do with them downstream — `instantiate(H∞)` is an `MPSKit.InfiniteMPOHamiltonian` — is
# on the [Handing the MPO to MPSKit](mpskit.md) page.

# ## What is not available here
#
# The SVD strategies are finite-only. It compresses each bond independently against a
# vacuum-terminated layout, and there is no bond basis that closes on itself, so there is nothing for
# the fixed point to converge to:

try
    mpo(opsum(cell1, ex[1, 2]), IndependentSVD(truncrank(2)))
catch e
    println(first(sprint(showerror, e), 160))
end

# Terms with no support (`K = 0`) are rejected too, on insertion: ``\sum_n c\,\mathbb{1}`` does not
# converge.
#
# Infinite-range couplings are a different matter — a *geometric* decay is representable exactly, at
# fixed cost, and has its own page: [Exponentially decaying interactions](exponential_decay.md).
