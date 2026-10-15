# # Fermions: free chains and the Hubbard model
#
# Fermionic models need no special support here: TensorKit's graded (fermionic) sectors carry the
# anticommutation through the braiding, and the letter machinery inherits it. In particular
# **no Jordan–Wigner strings appear anywhere** — the odd-parity charge flowing along the virtual
# bond does that bookkeeping for you.
#
# There is exactly one thing to get right — the anticommutation sign — and this page is built around
# it, including who supplies it.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## A fermionic mode
#
# [`fermion_basis`](@ref OpSum.fermion_basis)`(U1Irrep)` is a single spinless mode, empty or occupied,
# graded by particle number together with the fermion parity that
# makes the braiding fermionic (`FermionParity ⊠ U1Irrep`). Its builder gives the neutral operators
# the symmetry admits, named after TensorKitTensors.jl: `f_num` (``\hat n``), `f_plus_f_min` (``c^\dagger_i c_j``), `f_min_f_plus`
# (``c_i c^\dagger_j``) and `f_hopping` (the hermitian combination). The charged ``c^\dagger`` and ``c``
# are in the basis, but never appear in model code.

Bf = fermion_basis(U1Irrep)
F = fermion_ops(U1Irrep)
keys(F)

#-

BraidingStyle(sectortype(Bf))

# ## The sign, and who supplies it
#
# ```math
# H = -t \sum_i \left( c^\dagger_i c_{i+1} + c^\dagger_{i+1} c_i \right)
# ```
#
# A term is *stored* in site order, so `f_plus_f_min[i + 1, i]` (``c^\dagger_{i+1} c_i``, the
# creation operator on the *right*) cannot be kept as written: its two legs have to be swapped, and
# swapping two odd-parity legs costs
#
# ```math
# c^\dagger_{i+1} c_i = - c_i c^\dagger_{i+1}.
# ```
#
# Placement does that swap and inserts the phase itself, so the Hamiltonian is written the way it is
# printed above. `H'` produces the same partner from the first half — the one operation that needs
# the basis, since a letter's adjoint is a combination of the dual charge's letters, so it lives on
# the `OperatorSum` that `opsum(lat, …)` returns.

chain(N) = FiniteChain(Bf, N)

function hopping(N; t = 1.0)
    return opsum(
        chain(N),
        -t * (F.f_plus_f_min[i, i + 1] + F.f_plus_f_min[i + 1, i]) for i in 1:(N - 1)
    )
end

N = 8
H = hopping(N)

forward = opsum(chain(N), -1.0 * F.f_plus_f_min[i, i + 1] for i in 1:(N - 1))
H ≈ forward + forward'

# The builder's `f_hopping` is the same combination, in one name:

H ≈ opsum(chain(N), -1.0 * F.f_hopping[i, i + 1] for i in 1:(N - 1))

# Nothing stops you from supplying a sign by hand, and then it is yours to get right: written in
# storage order the h.c. partner of ``c^\dagger_i c_{i+1}`` is ``-c_i c^\dagger_{i+1}``, and a `+`
# there is not a subtle error — the operator simply stops being hermitian, which is a cheap and very
# sharp check.

byhand(N; t = 1.0, sign = -1.0) = opsum(
    chain(N),
    -t * (F.f_plus_f_min[i, i + 1] + sign * F.f_min_f_plus[i, i + 1]) for i in 1:(N - 1)
)

H ≈ byhand(N)
#-
(
    correct = hermiticity_error(H),
    wrong = hermiticity_error(byhand(N; sign = +1.0)),
)

# ## Verification against the exact spectrum
#
# For an open chain the single-particle energies are ``\varepsilon_k = -2t\cos(k\pi/(N+1))`` and the
# many-body spectrum is every subset sum of them. Note that we compare *eigenvalues*, computed block
# by block — for fermionic sectors `convert(Array, t)` is not a well-defined operation (TensorKit
# will warn that it does not preserve the categorical structure), so the dense-matrix oracle used
# elsewhere is simply unavailable here.

levels(N; t = 1.0) = [-2t * cos(k * π / (N + 1)) for k in 1:N]
subsetsums(ε) = sort([sum(ε[k] for k in eachindex(ε) if isodd(m >> (k - 1)); init = 0.0) for m in 0:(2^length(ε) - 1)])

spectrum(H) ≈ subsetsums(levels(N))

# The MPO is lossless, and its bond dimension is independent of ``N`` just as for the spin chains:

islossless(H)

res_free = build("free fermions", H)

# ## Adding a density–density interaction
#
# ```math
# H = -t \sum_i \left( c^\dagger_i c_{i+1} + \mathrm{h.c.} \right) + V \sum_i \hat{n}_i \hat{n}_{i+1}
# ```
#
# The ``t``–``V`` model. ``\hat n_i \hat n_{i+1}`` is the product of two placed operators.

function tV_chain(N; t = 1.0, Vint = 2.0)
    return opsum!(
        hopping(N; t),
        (Vint * F.f_num[i] * F.f_num[i + 1] for i in 1:(N - 1))
    )
end

H_tV = tV_chain(N)
res_tV = build("t-V chain", H_tV)
islossless(H_tV)

# ## Pairing: a symmetry that is not U(1)
#
# ```math
# H = \sum_i \left[ -t \left( c^\dagger_i c_{i+1} + \mathrm{h.c.} \right)
#     + \Delta \left( c^\dagger_i c^\dagger_{i+1} + \mathrm{h.c.} \right) \right]
# ```
#
# The Kitaev chain. Pairing creates two particles at once, so particle number is *not* conserved —
# only its parity. A `fermion_basis(U1Irrep)` site has no `f_plus_f_plus`, because the charges cannot
# fuse to the unit sector. Grade by `FermionParity` alone (the default `fermion_basis()`) and they can:
# `1 ⊗ 1 = 0`.
#
# This is worth being explicit about, because it is a choice the *model* makes and the package only
# follows: the symmetry you grade the site by is the symmetry you get to exploit.

Bp = fermion_basis()
Fp = fermion_ops()
sectortype(Bp)

# The builder now also has the pairing operators `f_plus_f_plus` (``c^\dagger_i c^\dagger_j``) and
# `f_min_f_min` (``c_i c_j``), which no number-conserving symmetry admits as neutral.

function kitaev(N; t = 1.0, Δ = 0.5)
    bonds = [(i, i + 1) for i in 1:(N - 1)]
    return opsum(
        FiniteChain(Bp, N),
        (-t * (Fp.f_plus_f_min[i, j] + Fp.f_plus_f_min[j, i]) for (i, j) in bonds),
        (Δ * (Fp.f_plus_f_plus[i, j] + Fp.f_min_f_min[j, i]) for (i, j) in bonds),
    )
end

H_kitaev = kitaev(N)
build("Kitaev chain", H_kitaev)
islossless(H_kitaev)

# The pairing terms are free: the same `D = 4` as the hopping chain alone, because the pair channel
# reuses the bond index the hopping channel already needs. And the signs are still nobody's problem
# but placement's — hermiticity is the sharpest check on them:

hermiticity_error(H_kitaev) < 1.0e-12

# At ``\Delta = t`` and zero chemical potential the chain is the Majorana fixed point; the spectrum
# is the honest check against any external reference:

spectrum(kitaev(6; t = 1.0, Δ = 1.0))[1:4]

# ## The Fermi–Hubbard model
#
# ```math
# H = -t \sum_{i,\sigma} \left( c^\dagger_{i\sigma} c_{i+1,\sigma} + \mathrm{h.c.} \right)
#     + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}
# ```
#
# There are two ways to put this on a chain, and the page does both. The first, here, gives **each
# spin-orbital its own site**: every site is a spinless mode, and the on-site interaction
# ``\hat{n}_{i\uparrow}\hat{n}_{i\downarrow}`` becomes a product of two operators on two sites, the two
# orbitals of the same physical site. We interleave them, ``(i,\sigma) \mapsto 2(i-1) + \sigma``, so
# each physical site is contiguous. The second, at the end of the page, keeps the four-dimensional
# Fock space of a physical site as one site, with the spin in the symmetry.

orbital(i, σ) = 2 * (i - 1) + σ    # σ = 1 for ↑, 2 for ↓

function hubbard(Nsites; t = 1.0, U = 4.0)
    hop = [
        -t * (
            F.f_plus_f_min[orbital(i, σ), orbital(i + 1, σ)] +
                F.f_plus_f_min[orbital(i + 1, σ), orbital(i, σ)]
        )
            for i in 1:(Nsites - 1) for σ in 1:2
    ]
    int = [
        U * F.f_num[orbital(i, 1)] * F.f_num[orbital(i, 2)]
            for i in 1:Nsites
    ]
    return opsum(chain(2Nsites), hop, int)
end

Nsites = 6
H_hub = hubbard(Nsites)
res_hub = build("Hubbard 1D", H_hub)
islossless(H_hub)

# ### Checking it against two exactly solvable limits
#
# At ``U = 0`` the model is two independent species of free fermions, so the spectrum is every
# subset sum over the two copies of the single-particle levels:

spectrum(hubbard(3; U = 0.0)) ≈ subsetsums(repeat(levels(3), 2))

# At ``t = 0`` nothing moves, so the energy just counts doubly-occupied sites in units of ``U``:

let Ns = 3, U = 4.0
    Ht = hubbard(Ns; t = 0.0, U)
    sort(unique(round.(spectrum(Ht); digits = 8)))
end

# ### Bond dimension
#
# As for every other local model, the Hubbard bond dimension saturates:

for Ns in (4, 6, 8, 12)
    r = build("hub", hubbard(Ns); quiet = true)
    println("  sites=$(rpad(Ns, 3)) orbitals=$(rpad(2Ns, 3))  D=$(rpad(r.D, 3))  D_dense=$(r.Ddense)")
end

# The spin-orbital encoding leaves a visible fingerprint on the *profile* of the bond dimension: it
# alternates between consecutive bonds, because a bond cutting *within* a physical site (between its
# ↑ and ↓ orbitals) carries the half-finished on-site interaction, while a bond cutting *between*
# physical sites does not.

let Ns = 8
    secs = mpo(hubbard(Ns)).bondsectors
    [densedim(secs, b) for b in eachindex(secs)]
end

# ## Hubbard on a cylinder
#
# Nothing about the construction is one-dimensional: swapping the chain's bond list for a cylinder's
# gives the quasi-2D Hubbard model. The physical sites form the cylinder, and each carries its two
# spin-orbitals.

function hubbard_cylinder(Lx, Ly; t = 1.0, U = 4.0)
    Nsites = Lx * Ly
    hop = [
        -t * (
            F.f_plus_f_min[orbital(i, σ), orbital(j, σ)] +
                F.f_plus_f_min[orbital(j, σ), orbital(i, σ)]
        )
            for (i, j) in cylinder_bonds(Lx, Ly) for σ in 1:2
    ]
    int = [U * F.f_num[orbital(i, 1)] * F.f_num[orbital(i, 2)] for i in 1:Nsites]
    return opsum(chain(2Nsites), hop, int)
end

for Lx in (2, 4, 6, 8)
    r = build("hub cyl", hubbard_cylinder(Lx, 4); quiet = true)
    println("  $(Lx)x4 cylinder: sites=$(rpad(4Lx, 3)) orbitals=$(rpad(8Lx, 3))  D_dense=$(r.Ddense)")
end

# The bond dimension saturates at 19 once the cylinder is longer than two columns, and stays there
# out to hundreds of sites — the same length-independence seen for the spin cylinders.

# ## The Hubbard site as a whole
#
# The other encoding keeps one site per physical site, with the four-dimensional Fock space
# (empty, ``\uparrow``, ``\downarrow``, ``\uparrow\downarrow``) and the symmetry chosen on the *basis*:
# [`hubbard_basis`](@ref OpSum.hubbard_basis)`(particle, spin)`. There ``c^\dagger_\uparrow`` is a
# single element of the basis, so `e_hopping` (the spin-summed hop ``\sum_\sigma c^\dagger_{i\sigma}
# c_{j\sigma} + \mathrm{h.c.}``) and `ud_num` (``\hat n_\uparrow\hat n_\downarrow``) are what the model needs.
#
# With the spin kept as a ``U(1)`` charge too (`hubbard_basis(U1Irrep, U1Irrep)`), and with SU(2) —
# the hardest case in the series, **non-abelian and fermionic at once**, graded by particle number,
# total spin and fermion parity:

function hubbard_site(Ns, P, S; t = 1.0, U = 4.0)
    ops = hubbard_ops(P, S)
    return opsum(
        FiniteChain(hubbard_basis(P, S), Ns),
        (-t * ops.e_hopping[i, i + 1] for i in 1:(Ns - 1)),
        (U * ops.ud_num[i] for i in 1:Ns),
    )
end

hubbard_u1(Ns; kwargs...) = hubbard_site(Ns, U1Irrep, U1Irrep; kwargs...)
hubbard_su2(Ns; kwargs...) = hubbard_site(Ns, U1Irrep, SU2Irrep; kwargs...)

H_hub2 = hubbard_su2(4)
build("Hubbard SU(2), N=4", H_hub2)
islossless(H_hub2)

# Operators written by hand deserve an independent check, and there is a good one available: the
# spin-orbital encoding above is the same physics through a completely different route — a mode per
# spin-orbital, nothing about spin in the symmetry. Comparing spectra is what makes the conventions
# of the stock operators trustworthy rather than merely transcribed.

let Nsites = 3
    a = spectrum(hubbard_su2(Nsites))
    b = spectrum(hubbard(Nsites))       # `hubbard` counts sites, its chain has one site per orbital
    c = spectrum(hubbard_u1(Nsites))
    (; dim = length(a), su2 = a ≈ b, u1 = c ≈ b)
end

# And the payoff for the extra symmetry, against the spin-orbital encoding of the same model:

let Nsites = 4
    su2 = build("Hubbard SU(2)", hubbard_su2(Nsites); quiet = true)
    u1 = build("Hubbard U(1)xU(1)", hubbard_u1(Nsites); quiet = true)
    orb = build("Hubbard spin-orbital", hubbard(Nsites); quiet = true)
    println("  SU(2) spin:     N=$Nsites sites   D=$(su2.D)  D_dense=$(su2.Ddense)")
    println("  U(1) spin:      N=$Nsites sites   D=$(u1.D)  D_dense=$(u1.Ddense)")
    println("  spin-orbital:   N=$(2Nsites) orbitals  D=$(orb.D)  D_dense=$(orb.Ddense)")
end

# The extra symmetry pays: at ``N = 4`` the SU(2) model needs 4 reduced indices (6 dense-equivalent)
# where the U(1)×U(1) one needs 6 and the spin-orbital encoding 7. Both single-site encodings are
# smaller than the spin-orbital one because the on-site interaction and the spin-summed hop are single
# letters of the Hubbard basis.
