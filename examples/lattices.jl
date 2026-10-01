# # Lattices and geometry
#
# Two things live in the lattice and nowhere else: **which space sits on each site**, and
# **how the sites are ordered**. The term algebra never sees either — it knows site indices and
# charges — so this page changes only the lattice the terms are put on, never how a term is written.
#
# Two consequences, one per half of the page:
#
#  * the sites need not carry the *same* space (`FiniteChain(V, N)` is only the convenient case);
#  * a two-dimensional lattice has to be *flattened*, and how far a bond spans after flattening is
#    what sets the bond dimension. For a nearest-neighbour Heisenberg cylinder it is exactly
#    ``D_\mathrm{dense} = 3 L_y + 2``, independent of ``L_x``.

using OpSum: OpSum
include(joinpath(pkgdir(OpSum), "examples", "common.jl"))

# ## A non-uniform chain
#
# `FiniteChain` takes one space per site. An alternating spin-½ / spin-1 chain is then just a list:

Va = SU2Space(1 // 2 => 1)
Vb = SU2Space(1 => 1)
alt = FiniteChain([Va, Vb, Va, Vb, Va, Vb])
length(alt)

# The operators are the ordinary ones — but `spin(V)` depends on the space, so a bond needs the
# right one at each end. That is the only bookkeeping a non-uniform lattice adds.

Sa, Sb = spin(Va), spin(Vb)
op(i) = isodd(i) ? Sa : Sb

function alternating(N; J = 1.0)
    spaces = [isodd(i) ? Va : Vb for i in 1:N]
    return opsum(FiniteChain(spaces), J * dot(op(i)[i], op(i + 1)[i + 1]) for i in 1:(N - 1))
end

build("alternating ½/1", alternating(6))

# The `OperatorSum` the builder returns carries its lattice, so there is a single value to pass
# around; the benchmark registry does the same.

islossless(alternating(6))

# Adding terms to a lattice is where operators and spaces are first confronted, so this is where a
# mismatch is caught. Putting the spin-1 operator on a spin-½ site is an error, not a wrong number:

try
    opsum(FiniteChain(Va, 2), dot(Sb[1], Sb[2]))
catch e
    println(sprint(showerror, e))
end

# ## An impurity
#
# The practical version of the same thing: one different site in an otherwise uniform chain. Nothing
# in the model code changes except which operator is used at that site.

function impurity(N, at; J = 1.0)
    spaces = [i == at ? Vb : Va for i in 1:N]
    o(i) = i == at ? Sb : Sa
    return opsum(FiniteChain(spaces), J * dot(o(i)[i], o(i + 1)[i + 1]) for i in 1:(N - 1))
end

for at in (1, 3, 6)
    Ws, secs = irrep_mpo(impurity(6, at))
    println("  impurity at $at:  per-bond D = ", [bonddim(secs, b) for b in eachindex(secs)])
end

# And the answer is that it costs **nothing**: the same per-bond dimensions as a uniform spin-½
# chain, wherever the impurity sits. Worth dwelling on, because the naive expectation is wrong. A
# bigger local space does not mean a bigger bond: what crosses a cut here is one open spin-1 channel
# either way, since `dot` couples through the spin-1 channel whatever the sites' spins are. The
# alternating chain above is the same story — `D = 3` there too.
#
# The local space shows up in the *dense-equivalent* figure only where the operator actually opens
# more channels, which for a nearest-neighbour Heisenberg coupling it never does. A biquadratic term
# on the spin-1 sites would (see the
# [bilinear-biquadratic section](spin_chains.md)).

# ## Flattening two dimensions
#
# An MPO lives on a chain, so a 2D lattice has to be ordered. Column-major,
# ``(x, y) \mapsto (x-1) L_y + y``, with ``x`` along the cylinder and ``y`` around it: each rung is
# then contiguous and every bond spans at most ``L_y`` sites. That bounded span is precisely what
# keeps the bond dimension finite.
#
# `cylinder_bonds` emits bonds with the smaller linear index first, because `couple` is strictly
# left-to-right.

cylinder_bonds(3, 3)

# A cylinder is periodic around ``y``; a ladder is the same construction with open boundaries. (For
# ``L_y = 2`` the periodic wrap would emit the single rung twice and silently double its coupling, so
# `cylinder_bonds` drops it and says so.)

ladder_bonds(3, 2)

# ## The Hamiltonian
#
# ```math
# H = J \sum_{\langle i j \rangle} \vec{S}_i \cdot \vec{S}_j
# ```
#
# summed over the bonds of the chosen geometry. Every bond is a plain SU(2) scalar product, so the
# geometry is the *only* thing that changes between one and two dimensions.

V = Va
S = Sa
chain(N) = FiniteChain(V, N)

heisenberg_bonds(lat, bonds; J = 1.0) = opsum(lat, J * dot(S[i], S[j]) for (i, j) in bonds)
heisenberg_cylinder(Lx, Ly; kwargs...) =
    heisenberg_bonds(chain(Lx * Ly), cylinder_bonds(Lx, Ly); kwargs...)
heisenberg_ladder(Lx, Ly = 2; kwargs...) =
    heisenberg_bonds(chain(Lx * Ly), ladder_bonds(Lx, Ly); kwargs...)

# ## A two-leg ladder

Lx = 6
H_ladder = heisenberg_ladder(Lx)
res_ladder = build("ladder 6x2", H_ladder)

islossless(H_ladder)

# The smallest non-trivial case, a single ``2 \times 2`` plaquette, is small enough to check against
# the dense operator directly:

mpo_matches_oracle(heisenberg_cylinder(2, 2))

# ## Growth in the circumference
#
# Sweeping the circumference at fixed length gives a clean linear law. Each extra ring adds one more
# leg bond that can straddle a cut, and each open bond carries a spin-1 multiplet of quantum
# dimension 3 — hence ``3 L_y + 2``, the ``+2`` being the identity-in and identity-out channels.

for Ly in 3:6
    r = build("cylinder 4x$Ly", heisenberg_cylinder(4, Ly); quiet = true)
    println("  Ly=$Ly   D=$(rpad(r.D, 3))  D_dense=$(rpad(r.Ddense, 4))  3Ly+2 = $(3Ly + 2)")
end

# ## ... and independence from the length
#
# Stretching the cylinder at fixed circumference changes nothing. This is the whole reason cylinder
# DMRG is feasible: cost is exponential in circumference but only linear in length.

for Ly in (3, 4)
    for Lx in (3, 4, 6, 8)
        r = build("cyl", heisenberg_cylinder(Lx, Ly); quiet = true)
        println("  Ly=$Ly  Lx=$(rpad(Lx, 2))  N=$(rpad(Lx * Ly, 3))  D_dense=$(r.Ddense)")
    end
end

# Stated as an assertion over the whole grid:

all(
    build("c", heisenberg_cylinder(Lx, Ly); quiet = true).Ddense == 3Ly + 2
        for Ly in 3:6, Lx in (3, 4, 5)
)

# ## Symmetry reduction
#
# The reduced bond dimension — the number of symmetry-resolved indices, and the quantity a DMRG
# sweep actually pays for — is ``L_y + 2``: one index per open spin-1 multiplet plus the two identity
# channels. Imposing SU(2) therefore shrinks the bond by roughly a factor of three relative to a
# symmetry-agnostic MPO of the same operator.

map(3:6) do Ly
    r = build("c", heisenberg_cylinder(4, Ly); quiet = true)
    return (Ly = Ly, reduced = r.D, dense = r.Ddense, ratio = round(r.Ddense / r.D; digits = 2))
end

# ## Scaling
#
# ```
# julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full
# ```
#
# ![Bond dimension and construction time versus system size](../assets/scaling.png)
#
# Along the chain the bond dimension rises over the first ``L_y`` sites — the length of one ring —
# and then sits on a flat plateau however long the cylinder is:
#
# ![Bond dimension profile along the chain](../assets/profile.png)
