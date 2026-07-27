# # Fermions: free chains and the Hubbard model
#
# Fermionic models need no special support here: TensorKit's graded (fermionic) sectors carry the
# anticommutation through the braiding, and the ITO machinery inherits it. In particular
# **no Jordan–Wigner strings appear anywhere** — the odd-parity charge flowing along the virtual
# bond does that bookkeeping for you.
#
# There is exactly one thing you have to get right, and this page is built around it.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## A fermionic mode
#
# `FermionNumber` is `U1Irrep ⊠ FermionParity`: particle number together with the parity that makes
# the braiding fermionic. A single spinless mode is empty or occupied:

V = Vect[FermionNumber](0 => 1, 1 => 1)
vac, occ = FermionNumber(0), FermionNumber(1)

# The operators are matrix units, derived rather than hard-coded:

c = matrixunit(V, vac, occ)    # annihilation, ``c``
cd = matrixunit(V, occ, vac)   # creation, ``c^\dagger``
nh = matrixunit(V, occ, occ)   # number, ``\hat{n}``

BraidingStyle(sectortype(V))

# ## The one sign you have to get right
#
# ```math
# H = -t \sum_i \left( c^\dagger_i c_{i+1} + c^\dagger_{i+1} c_i \right)
# ```
#
# `couple` builds its fusion tree strictly left to right, so there is no way to write
# ``c^\dagger_{i+1} c_i`` directly — the operator on the *smaller* site index must come first.
# Anticommuting it into place costs a sign:
#
# ```math
# c^\dagger_{i+1} c_i = - c_i c^\dagger_{i+1}
# ```
#
# so the hermitian conjugate partner enters with a **minus**.

function hopping(N; t = 1.0, sign = -1.0)
    return -t * sum(
        [
            couple(cd[i], c[i + 1]) + sign * couple(c[i], cd[i + 1])
                for i in 1:(N - 1)
        ]
    )
end

N = 8
sites = fill(V, N)
H = hopping(N)

# Getting the sign wrong is not a subtle error — the operator simply stops being hermitian, which
# is a cheap and very sharp check:

(correct = hermiticity_error(H, sites), wrong = hermiticity_error(hopping(N; sign = +1.0), sites))

# ## Verification against the exact spectrum
#
# For an open chain the single-particle energies are ``\varepsilon_k = -2t\cos(k\pi/(N+1))`` and the
# many-body spectrum is every subset sum of them. Note that we compare *eigenvalues*, computed block
# by block — for fermionic sectors `convert(Array, t)` is not a well-defined operation (TensorKit
# will warn that it does not preserve the categorical structure), so the dense-matrix oracle used
# elsewhere is simply unavailable here.

function free_fermion_spectrum(N; t = 1.0)
    ε = [-2t * cos(k * π / (N + 1)) for k in 1:N]
    return sort(
        [
            sum(ε[k] for k in 1:N if (m >> (k - 1)) & 1 == 1; init = 0.0)
                for m in 0:(2^N - 1)
        ]
    )
end

spectrum(H, sites) ≈ free_fermion_spectrum(N)

# The MPO is lossless, and its bond dimension is independent of ``N`` just as for the spin chains:

islossless(H, sites)

res_free = build("free fermions", H, sites)

# ## Adding a density–density interaction
#
# ```math
# H = -t \sum_i \left( c^\dagger_i c_{i+1} + \mathrm{h.c.} \right) + V \sum_i \hat{n}_i \hat{n}_{i+1}
# ```
#
# The ``t``–``V`` model. Both ``\hat{n}`` factors are single letters, so this is an ordinary
# two-site term.

function tV_chain(N; t = 1.0, Vint = 2.0)
    return hopping(N; t) + Vint * sum([couple(nh[i], nh[i + 1]) for i in 1:(N - 1)])
end

H_tV = tV_chain(N)
res_tV = build("t-V chain", H_tV, sites)
islossless(H_tV, sites)

# ## The Fermi–Hubbard model
#
# ```math
# H = -t \sum_{i,\sigma} \left( c^\dagger_{i\sigma} c_{i+1,\sigma} + \mathrm{h.c.} \right)
#     + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}
# ```
#
# A spinful site has four states, and there ``c^\dagger_\uparrow`` is unavoidably a *sum* of two
# alphabet letters (it maps both ``|0\rangle \to |\uparrow\rangle`` and
# ``|\downarrow\rangle \to |\uparrow\downarrow\rangle``) whose relative sign is a basis convention.
#
# The standard way to sidestep that entirely is to give **each spin-orbital its own site**. Then
# every operator is a single letter again, and the on-site interaction
# ``\hat{n}_{i\uparrow}\hat{n}_{i\downarrow}`` becomes an ordinary *two-site* term between the two
# orbitals of the same physical site. We interleave them, ``(i,\sigma) \mapsto 2(i-1) + \sigma``, so
# each physical site is contiguous.

orbital(i, σ) = 2 * (i - 1) + σ    # σ = 1 for ↑, 2 for ↓

function hubbard(Nsites; t = 1.0, U = 4.0)
    hop = [
        -t * (
                couple(cd[orbital(i, σ)], c[orbital(i + 1, σ)]) -
                couple(c[orbital(i, σ)], cd[orbital(i + 1, σ)])
            )
            for i in 1:(Nsites - 1) for σ in 1:2
    ]
    int = [
        U * couple(nh[orbital(i, 1)], nh[orbital(i, 2)])
            for i in 1:Nsites
    ]
    return sum(vcat(hop, int)), fill(V, 2Nsites)
end

Nsites = 6
H_hub, sites_hub = hubbard(Nsites)
res_hub = build("Hubbard 1D", H_hub, sites_hub)
islossless(H_hub, sites_hub)

# ### Checking it against two exactly solvable limits
#
# At ``U = 0`` the model is two independent species of free fermions, so the spectrum is every
# subset sum over the two copies of the single-particle levels:

let Ns = 3
    H0, s0 = hubbard(Ns; U = 0.0)
    ε = vcat(
        [-2 * cos(k * π / (Ns + 1)) for k in 1:Ns],
        [-2 * cos(k * π / (Ns + 1)) for k in 1:Ns],
    )
    exact = sort(
        [
            sum(ε[k] for k in 1:(2Ns) if (m >> (k - 1)) & 1 == 1; init = 0.0)
                for m in 0:(2^(2Ns) - 1)
        ]
    )
    spectrum(H0, s0) ≈ exact
end

# At ``t = 0`` nothing moves, so the energy just counts doubly-occupied sites in units of ``U``:

let Ns = 3, U = 4.0
    Ht, st = hubbard(Ns; t = 0.0, U)
    sort(unique(round.(spectrum(Ht, st); digits = 8)))
end

# ### Bond dimension
#
# As for every other local model, the Hubbard bond dimension saturates:

for Ns in (4, 6, 8, 12)
    H, s = hubbard(Ns)
    r = build("hub", H, s; quiet = true)
    println("  sites=$(rpad(Ns, 3)) orbitals=$(rpad(2Ns, 3))  D=$(rpad(r.D, 3))  D_dense=$(r.Ddense)")
end

# The spin-orbital encoding leaves a visible fingerprint on the *profile* of the bond dimension: it
# alternates between consecutive bonds, because a bond cutting *within* a physical site (between its
# ↑ and ↓ orbitals) carries the half-finished on-site interaction, while a bond cutting *between*
# physical sites does not.

let Ns = 8
    H, s = hubbard(Ns)
    _, secs = irrep_mpo(H, s)
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
                couple(cd[orbital(i, σ)], c[orbital(j, σ)]) -
                couple(c[orbital(i, σ)], cd[orbital(j, σ)])
            )
            for (i, j) in cylinder_bonds(Lx, Ly) for σ in 1:2
    ]
    int = [U * couple(nh[orbital(i, 1)], nh[orbital(i, 2)]) for i in 1:Nsites]
    return sum(vcat(hop, int)), fill(V, 2Nsites)
end

for Lx in (2, 4, 6, 8)
    H, s = hubbard_cylinder(Lx, 4)
    r = build("hub cyl", H, s; quiet = true)
    println("  $(Lx)x4 cylinder: sites=$(rpad(4Lx, 3)) orbitals=$(rpad(8Lx, 3))  D_dense=$(r.Ddense)")
end

# The bond dimension saturates at 19 once the cylinder is longer than two columns, and stays there
# out to hundreds of sites — the same length-independence seen for the spin cylinders.

# ## Scaling
#
# ```
# julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full
# ```
#
# ![Bond dimension and construction time versus system size](../assets/scaling.png)
