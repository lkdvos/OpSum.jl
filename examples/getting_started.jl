# # Getting started
#
# One bond, then one Hamiltonian, with every step named. Every later page adds exactly one thing to
# what is here.
#
# The pipeline has three objects:
#
#  1. a **term bag** — what you write down. It knows site indices and charges, and **no lattice**.
#  2. an **`OperatorSum`** — the term bag together with a **lattice** (one physical space per site).
#     The terms are checked against the spaces as they go in, which is the first place the spaces
#     are actually needed.
#  3. an **MPO** — the compressed result.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## One bond
#
# `spin(V)` is the single irreducible tensor operator of an SU(2) site — the vector operator
# ``\vec{S}`` itself, not three separate components. `S[i]` places it on site `i`, and `dot` couples
# two placements to a scalar.

V = SU2Space(1 // 2 => 1)
S = spin(V)
h = dot(S[1], S[2])

# That is already a `Terms`: a bag of terms. It carries no lattice, so it does not yet know what a
# "site" is physically — only that there are two of them and which charges they carry. Putting it
# on a lattice gives an `OperatorSum`, which is what the MPO is formed from:

lat = FiniteChain(V, 2)
H2 = opsum(lat, h)
mpo = irrep_mpo(H2)

# The MPO destructures as the pair it holds, on either lattice:

Ws, secs = mpo
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

mpo_terms(Ws, secs) ≈ h

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

heisenberg(N; J = 1.0) = opsum(FiniteChain(V, N), J * dot(S[i], S[i + 1]) for i in 1:(N - 1))
H = heisenberg(8)
build("Heisenberg N=8", H)

# The bond dimension does not grow with the chain — the model is finite-range, so each cut has the
# same amount of information crossing it:

for N in (8, 16, 64, 256)
    build("Heisenberg N=$N", heisenberg(N))
end

# ## No symmetry at all: transverse-field Ising
#
# `ℂ^2` is an ungraded space, so there is nothing to exploit and the Pauli matrices are simply
# written down. [`project`](@ref OpSum.project) expands a `TensorMap` in the on-site alphabet, which
# is how every on-site operator in these pages is built — never by writing a letter index by hand,
# because those follow TensorKit's block order and hard-coding one is a silent bug.

Vc = ℂ^2
σx = project(TensorMap(ComplexF64[0 1; 1 0], Vc ← Vc), Vc)
σz = project(TensorMap(ComplexF64[1 0; 0 -1], Vc ← Vc), Vc)

# This model also mixes arities: the field is a one-site term and the coupling a two-site one, in
# one operator. Nothing special is needed for that — `opsum` takes both.

function tfim(N; J = 1.0, g = 0.5)
    return opsum(
        FiniteChain(Vc, N),
        (-J * couple(σz[i], σz[i + 1]) for i in 1:(N - 1)),
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
# Grade the site by magnetisation and U(1) is available. `matrixunit(V, out, in)` is ``|out⟩⟨in|``,
# which is all the spin operators are:

Vu = Rep[U₁](0 => 1, 1 => 1)
up, dn = U1Irrep(1), U1Irrep(0)
Sp = matrixunit(Vu, up, dn)
Sm = matrixunit(Vu, dn, up)
Sz = (matrixunit(Vu, up, up) - matrixunit(Vu, dn, dn)) / 2

# `Sᶻ` is worth a second look: it is a **composite** on-site operator, two alphabet letters with
# coefficients, and ordinary arithmetic on `SiteOperator`s reads the way you would write it on
# paper. `couple` distributes over both operands, so nothing has to be expanded by hand.

length(Sz)

# [`spin_ops`](@ref OpSum.spin_ops) bundles exactly these three. Its `sectors` argument is in
# **descending** magnetisation, and is not optional: a `Vect[U₁]` spin site is as often labelled by
# particle number as by ``m``, and inferring which would be a silent guess.

spin_ops(Vu, up, dn).Sz ≈ Sz

# ```math
# H = \sum_i \tfrac12 \left( S^+_i S^-_{i+1} + S^-_i S^+_{i+1} \right) + \Delta\, S^z_i S^z_{i+1}
# ```

function xxz(N; Δ = 1.0)
    return opsum(
        FiniteChain(Vu, N),
        couple(Sp[i], Sm[i + 1]) / 2 + couple(Sm[i], Sp[i + 1]) / 2 +
            Δ * couple(Sz[i], Sz[i + 1]) for i in 1:(N - 1)
    )
end

Hxxz = xxz(6)
build("XXZ Δ=1 N=6", Hxxz)

# At ``Δ = 1`` this *is* the Heisenberg chain, just graded by U(1) instead of SU(2) — same operator,
# two symmetries. The MPOs are not comparable (different spaces, different bond labels), but the
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
# * [Spin chains](spin_chains.md) — non-abelian coupling, `dot` versus `couple`, and projecting a whole bond.
# * [Multi-body interactions](multibody.md) — three and four sites, and the fusion channels between them.
# * [Fermions](fermions.md) — who supplies the anticommutation sign.
# * [Lattices and geometry](lattices.md) — non-uniform chains, ladders, cylinders.
# * [Handing the MPO to MPSKit](mpskit.md) — DMRG on what you just built.
