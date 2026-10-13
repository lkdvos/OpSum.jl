# # Multi-body interactions
#
# Everything so far has been two-body. Products of operators are not limited to pairs: `*` between
# placed operators is the operator product, in the order written, so three- and four-body
# SU(2)-invariant interactions are written as products of the two-body ones that physics suggests —
# ``(\vec S_1\!\cdot\!\vec S_2)(\vec S_2\!\cdot\!\vec S_3)``, ``(\vec S_1\!\cdot\!\vec S_2)(\vec S_3\!\cdot\!\vec S_4)``.
# The fusion channels between the sites are inside the basis, so nothing has to be named.
#
# This page shows how products behave, which independent operators exist, and what they cost in bond
# dimension.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

B = spin_basis(SU2Irrep)
ex = spin_ops(SU2Irrep).S_exchange
chain(N) = FiniteChain(B, N)

# ## Three-body terms
#
# Two bonds that share a site multiply into a three-site operator. The shared site is an on-site
# *product*, which is resolved in the basis of that site when the term enters the `OperatorSum`, so
# the (complete) spin-½ basis always contains the answer:

H3 = opsum(chain(3), ex[1, 2] * ex[2, 3])
build("K=3 (S₁·S₂)(S₂·S₃)", H3)

# The factors do not commute, so the order matters, and the hermitian conjugate reverses it:

H3' ≈ opsum(chain(3), ex[2, 3] * ex[1, 2])

# Their commutator is the scalar spin chirality, ``\left[\vec S_1\!\cdot\!\vec S_2,
# \vec S_2\!\cdot\!\vec S_3\right] = i\, \vec S_1\!\cdot\!(\vec S_2\times\vec S_3)``, a genuinely
# three-body operator that is anti-hermitian (so ``i`` times it is the hermitian chirality). The
# check is against the dense operator, which is small here:

chirality = opsum(chain(3), ex[1, 2] * ex[2, 3] - ex[2, 3] * ex[1, 2])
instantiate(chirality') ≈ -instantiate(chirality)

# ## Four-body terms
#
# With four sites there are three ways to pair them up, and so three products of two scalar products:

pairings = [
    opsum(chain(4), ex[1, 2] * ex[3, 4]),
    opsum(chain(4), ex[1, 3] * ex[2, 4]),
    opsum(chain(4), ex[1, 4] * ex[2, 3]),
]
for p in pairings
    build("pairing", p)
end

# Placement may be in any site order, so the interleaved ``(1,3)(2,4)`` pairing is written exactly as
# it reads; under a non-abelian symmetry the legs are re-coupled into site order for you (which can
# make one written term several stored ones). Whether they are really three *different* four-body
# operators is a question about the dense operators, and the answer is in their overlap matrix:

let ops = [instantiate(p) for p in pairings]
    G = [real(dot(a, b) / (norm(a) * norm(b))) for a in ops, b in ops]
    (; overlaps = round.(G; digits = 6), rank = rank(G))
end

# They are linearly independent but not orthogonal (any two overlap by ``1/3``), and the three-site
# structure shows in the cost: the pairing of neighbours has one bond index everywhere, the other two
# need three. Three is exactly the number of independent singlet couplings of four spin-1 operators,
# which is what ``\vec S_i`` carries.

# ### A note on ring exchange
#
# The physical ring-exchange operator of four spins on a plaquette is a particular linear
# combination of these products: with the ring running ``a \to b \to c \to d``, it is the two
# pairings of *adjacent* sites minus the pairing of the *diagonals*,
#
# ```math
# (\vec S_a\!\cdot\!\vec S_b)(\vec S_c\!\cdot\!\vec S_d)
# + (\vec S_a\!\cdot\!\vec S_d)(\vec S_b\!\cdot\!\vec S_c)
# - (\vec S_a\!\cdot\!\vec S_c)(\vec S_b\!\cdot\!\vec S_d).
# ```
#
# No recoupling coefficients are needed: `*` and placement do it. The cost of a ring exchange is
# that of its most expensive pairing, which is what the ladder below measures.

# ## Four fermions: the same, and the sign is free
#
# The same idea holds for any symmetry, and the sign of a fermionic product is part of it. A
# four-fermion term is a product of two hops, ``c^\dagger_1 c_2\, c^\dagger_3 c_4``:

F = fermion_ops()
ring4 = F.f_plus_f_min[1, 2] * F.f_plus_f_min[3, 4]
length(ring4)

# Each hop is bosonic, so two hops commute, even when their sites interleave and the legs have to
# pass each other. `*` and placement carry the braiding sign of every leg that passes another, which
# is the whole Jordan–Wigner bookkeeping, and the check that they got it right is that the order of
# the factors does not matter:

let lat = FiniteChain(fermion_basis(), 4), A = F.f_plus_f_min[1, 3], Bhop = F.f_plus_f_min[2, 4]
    opsum(lat, A * Bhop) ≈ opsum(lat, Bhop * A)
end

# Read back off the term, the forced channels are the running fermion parity along the chain —
# odd after the first ``c^\dagger``, back to even after the first ``c``, and so on:

using OpSum: bondcharges, tree
bondcharges(tree(only(ring4)))

# ## A four-body model on a ladder
#
# To see multi-body terms in a Hamiltonian rather than in isolation, add ring-exchange terms to a
# two-leg ladder. With column-major ordering, rung ``x`` is sites ``2x-1, 2x``, so the four sites of
# the plaquette between rungs ``2k-1`` and ``2k`` are the consecutive indices ``4k-3, \ldots, 4k``
# (rungs `(4k-3, 4k-2)` and `(4k-1, 4k)`, legs `(4k-3, 4k-1)` and `(4k-2, 4k)`). The ring runs
# ``4k-3 \to 4k-2 \to 4k \to 4k-1``, so the diagonals are `(4k-3, 4k)` and `(4k-2, 4k-1)`.

function plaquette_ladder(Lx; J = 1.0, K = 0.3)
    bonds = ladder_bonds(Lx, 2)
    two_body = [J * ex[i, j] for (i, j) in bonds]
    plaquettes = [
        K * (
            ex[4k - 3, 4k - 2] * ex[4k - 1, 4k] +      # rungs
                ex[4k - 3, 4k - 1] * ex[4k - 2, 4k] -      # ... legs
                ex[4k - 3, 4k] * ex[4k - 2, 4k - 1]        # diagonals
        ) for k in 1:(div(Lx, 2))
    ]
    return opsum(chain(2Lx), two_body, plaquettes)
end

H4 = plaquette_ladder(6)
res4 = build("ladder + plaquettes", H4)
islossless(H4)

# Compare against the pure two-body ladder to isolate what the four-body terms cost:

let Lx = 6
    two_only = opsum(chain(2Lx), ex[i, j] for (i, j) in ladder_bonds(Lx, 2))
    two = build("ladder only", two_only; quiet = true)
    println("  two-body only:        D=$(two.D)  D_dense=$(two.Ddense)")
    println("  with plaquettes:      D=$(res4.D)  D_dense=$(res4.Ddense)")
end

# And the bond dimension still saturates with system size:

for Lx in (4, 6, 8, 12)
    r = build("plaq", plaquette_ladder(Lx); quiet = true)
    println("  Lx=$(rpad(Lx, 3)) N=$(rpad(2Lx, 3))  D=$(rpad(r.D, 3))  D_dense=$(r.Ddense)")
end
