# # Getting started
#
# One bond, then one Hamiltonian, with every step named. Every later page adds exactly one thing to
# what is here.
#
# The pipeline has three objects:
#
#  1. a **term bag** — what you write down. It knows site indices and charges, and **no lattice**.
#  2. an **`OperatorSum`** — the term bag together with a **lattice** (one operator basis per site).
#     The terms are checked against the bases as they go in, which is the first place the
#     physical sites are actually needed.
#  3. an **MPO** — the compressed result.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## One bond
#
# A site is a [`Basis`](@ref OpSum.Basis): the operators one can write on it, in the language of its
# symmetry. `spin_basis(SU2Irrep)` is the spin-½ site with SU(2) symmetry, and `spin_ops(SU2Irrep)`
# its operators expanded in that basis. Under SU(2) the only neutral spin operator on two sites is
# ``\vec S_i \!\cdot\! \vec S_j``: it is `S_exchange`, and `S_exchange[i, j]` places it on sites `i`
# and `j`.

B = spin_basis(SU2Irrep)
ops = spin_ops(SU2Irrep)
h = ops.S_exchange[1, 2]

# That is already a `Terms`: a bag of terms. It carries no lattice, so it does not yet know what a
# "site" is physically — only that there are two of them and which charges they carry. Putting it
# on a lattice, one basis per site, gives an `OperatorSum`, which is what the MPO is formed from:

lat = FiniteChain(B, 2)
H2 = opsum(lat, h)
W = mpo(H2)

# A [`SymbolicMPO`](@ref OpSum.SymbolicMPO) holds, per site, a sparse matrix of on-site operators
# (`W[i]`, or `W.Ws`), and per bond the charges of its indices (`W.bondsectors`), on either lattice:

Ws, secs = W.Ws, W.bondsectors
length(secs)

# ## Three ways to check it
#
# In increasing cost, and every later page uses at least the first.
#
# **Faithfulness.** Reconstruct the term bag from the bond data and compare. This is purely
# symbolic, needs no spaces — the bond data names charges but not spaces — and works at any
# `N` and for any sector, fermions included.

islossless(H2)

# Equivalently, spelled out: [`mpo_terms`](@ref OpSum.mpo_terms) is the inverse of the compression.

mpo_terms(W) ≈ h

# **The assembled tensors.** Build the symmetric `TensorMap`s, contract the chain, and compare
# against the dense oracle. Exponential in `N`, so small systems only, but it never leaves TensorKit
# and so is valid for fermionic sectors too.

mpo_matches_oracle(H2)

# **The spectrum.** The physics check, and the only honest one against an external reference. A
# single SU(2) bond has the singlet/triplet split ``-3/4`` and ``1/4``:

spectrum(H2)

# ## A Hamiltonian
#
# `opsum(lat, terms...)` accumulates any number of arguments in **one pass**, and iterables nest.
# `+` also works, but it copies, so folding it over many terms is quadratic — `opsum` (or `opsum!`
# into an `OperatorSum` you own) is the linear route.

heisenberg(N; J = 1.0) = opsum(FiniteChain(B, N), J * ops.S_exchange[i, i + 1] for i in 1:(N - 1))
H = heisenberg(8)
build("Heisenberg N=8", H)

# The bond dimension does not grow with the chain — the model is finite-range, so each cut has the
# same amount of information crossing it:

for N in (8, 16, 64, 256)
    build("Heisenberg N=$N", heisenberg(N))
end

# ## No symmetry at all: transverse-field Ising
#
# With `spin_basis()` (the default symmetry is `Trivial`) there is nothing to exploit, and the
# operators are the Cartesian ones. If the one you want is not in the builder, write it down and let
# [`project`](@ref OpSum.project)`(O, B)` expand it in the basis — never by writing a letter index by
# hand, because those follow TensorKit's block order and hard-coding one is a silent bug.

Bc = spin_basis()
Vc = space(Bc)
σx = project(TensorMap(ComplexF64[0 1; 1 0], Vc ← Vc), Bc)
σz = project(TensorMap(ComplexF64[1 0; 0 -1], Vc ← Vc), Bc)

# This model also mixes arities: the field is a one-site term and the coupling a two-site one, in
# one operator. Nothing special is needed for that — `opsum` takes both. Operators multiply with
# `*`, so the coupling is written the way it is printed:

function tfim(N; J = 1.0, g = 0.5)
    return opsum(
        FiniteChain(Bc, N),
        (-J * σz[i] * σz[i + 1] for i in 1:(N - 1)),
        (-g * σx[i] for i in 1:N),
    )
end

Htfim = tfim(6)
build("TFIM N=6", Htfim)
islossless(Htfim)

# At ``g = J`` the chain is critical; away from it the gap opens. The ground-state energy is the
# first entry of the spectrum:

for g in (0.0, 0.5, 1.0, 2.0)
    E = spectrum(tfim(6; g))[1]
    println("  g=$g  E₀=", round(E; digits = 6))
end

# ## An abelian symmetry: XXZ
#
# Grade the site by magnetisation and U(1) is available. The builder now has the three neutral
# operators a conserved ``S^z`` allows: `S_z`, and the two flip-flop products `S_plus_S_min` and
# `S_min_S_plus`, which are ``S^+_i S^-_j`` and ``S^-_i S^+_j``. The charged ``S^\pm`` are inside the
# basis, so the Hamiltonian

# ```math
# H = \sum_i \tfrac12 \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right) + \Delta\, S^z_i S^z_{i+1}
# ```

# is written as printed. `S_z[i] * S_z[i + 1]` is the operator product of two placed operators, and
# is the same operator as the builder's `S_z_S_z[i, i + 1]`:

Bu = spin_basis(U1Irrep)
ou = spin_ops(U1Irrep)

function xxz(N; Δ = 1.0)
    return opsum(
        FiniteChain(Bu, N),
        ou.S_plus_S_min[i, i + 1] / 2 + ou.S_min_S_plus[i, i + 1] / 2 +
            Δ * ou.S_z[i] * ou.S_z[i + 1] for i in 1:(N - 1)
    )
end

Hxxz = xxz(6)
build("XXZ Δ=1 N=6", Hxxz)

# At ``Δ = 1`` this *is* the Heisenberg chain, just graded by U(1) instead of SU(2) — same operator,
# two symmetries. The MPOs are not comparable (different bases, different bond labels), but the
# spectra are, once multiplet degeneracies are unfolded:

spectrum(Hxxz) ≈ spectrum(heisenberg(6))

# And this is the payoff of the symmetry, in one line: the same operator costs a smaller bond
# dimension under the larger group.

let N = 6
    u1 = build("XXZ (U(1))", xxz(N); quiet = true)
    su2 = build("Heisenberg (SU(2))", heisenberg(N); quiet = true)
    println("  U(1):   D=$(u1.D)  D_dense=$(u1.Ddense)")
    println("  SU(2):  D=$(su2.D)  D_dense=$(su2.Ddense)")
end

# ## Where to go next
#
# * [Spin chains](spin_chains.md) — the stock spin sites, products of operators, and projecting a whole bond.
# * [Multi-body interactions](multibody.md) — products of three and four operators, and their placement.
# * [Fermions](fermions.md) — who supplies the anticommutation sign.
# * [Lattices and geometry](lattices.md) — non-uniform chains, ladders, cylinders.
# * [Handing the MPO to MPSKit](mpskit.md) — DMRG on what you just built.
