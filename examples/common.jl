# # Shared utilities
#
# Every example page begins by `include`ing this file. It contains no physics — only the
# scaffolding that the examples share:
#
#  1. quasi-2D lattice geometry,
#  2. reporting of bond dimensions and construction times,
#  3. three tiers of verification.
#
# The file is plain Julia, so the same `include` works whether you are reading the rendered docs
# or running an example directly from a checkout.

using OpSum
using TensorKit
using LinearAlgebra: dot, eigvals, norm

# ## Writing down operators
#
# A site is a [`Basis`](@ref OpSum.Basis): a list of operators on the physical space, the identity
# first, in which every on-site operator is expanded. You rarely build one by hand. Each common site
# has a *stock basis* and a *builder* that hands out its operators already expanded in it, named
# after the functions of TensorKitTensors.jl:
#
# | site | basis | operators |
# |:--|:--|:--|
# | spin ``s`` | `spin_basis(sym; spin)` | `spin_ops(sym; spin)`: `S_z`, `S_plus_S_min`, `S_exchange`, … |
# | spinless fermion | `fermion_basis(sym)` | `fermion_ops(sym)`: `f_num`, `f_plus_f_min`, `f_hopping`, … |
# | boson | `boson_basis(sym; cutoff)` | `boson_ops(sym; cutoff)` |
# | Hubbard | `hubbard_basis(particle, spin)` | `hubbard_ops(particle, spin)`: `e_num`, `ud_num`, `e_plus_e_min`, … |
# | ``t``–``J`` | `tj_basis(particle, spin)` | `tj_ops(particle, spin)` |
#
# The builders return a `NamedTuple` of every *charge-neutral* operator of the site that the symmetry
# admits. A single-site entry (`ops.S_z`) is placed with `ops.S_z[i]`; a two-site entry
# (`ops.S_exchange`, ``\vec S_i\!\cdot\!\vec S_j``) is an unplaced [`LocalOperator`](@ref
# OpSum.LocalOperator), placed with `ops.S_exchange[i, j]`. The charged pieces a symmetry needs (the
# spin-1 operator ``\vec S`` of SU(2), the ``c^\dagger`` of a fermion) live *inside* the basis: you
# never write one down. Three consequences are worth having in mind for the pages that follow:
#
#  * the operators are neutral, so a Hamiltonian is written the way it is printed. Products are the
#    operator product, in the order written, and placement may be in any site order: for fermions
#    `hop[j, i]` is `c†ⱼ cᵢ`, with its anticommutation sign;
#  * for any other operator, write it as a `TensorMap` and expand it with
#    [`project`](@ref OpSum.project)`(O, B)`. Letter indices follow TensorKit's block order and are
#    never hard-coded;
#  * nothing is ever converted to a dense array, which matters for fermionic sectors where
#    `convert(Array, t)` is not a well-defined operation.
#
# `H'` gives the hermitian-conjugate partner of an [`OperatorSum`](@ref OpSum.OperatorSum)
# (`opsum(lat, terms...)`), the one place operators and sites meet, where letters are also checked
# against the basis of the site they sit on.

# ## Mixed-spin bonds
#
# The builders give the exchange of two sites of the *same* spin. For two different spins, under SU(2)
# the coupling is diagonal in the total spin ``J`` of the pair, with
# ``\vec S_a\!\cdot\!\vec S_b = \tfrac12\left[J(J+1) - s_a(s_a+1) - s_b(s_b+1)\right]``: one number
# per coupled sector of a `TensorMap`, which `project` expands in the list of the two bases.

"""
    exchange(Ba, Bb) -> LocalOperator

S⃗ₐ·S⃗_b between SU(2) sites with bases `Ba` and `Bb`, unplaced: `exchange(Ba, Bb)[i, j]`.
"""
function exchange(Ba, Bb)
    Va, Vb = space(Ba), space(Bb)
    sa, sb = only(sectors(Va)).j, only(sectors(Vb)).j
    t = zeros(ComplexF64, Va ⊗ Vb ← Va ⊗ Vb)
    for (J, blk) in blocks(t)
        blk .= (J.j * (J.j + 1) - sa * (sa + 1) - sb * (sb + 1)) / 2
    end
    return project(t, [Ba, Bb])
end

# ## Quasi-2D geometry
#
# An MPO lives on a chain, so a two-dimensional lattice has to be *ordered*. We use column-major
# ordering, `(x, y) ↦ (x-1)*Ly + y`, with `x` running along the cylinder and `y` around it. Each
# rung is then contiguous and every bond spans at most `Ly` sites — that locality is exactly what
# keeps the MPO bond dimension linear in the circumference `Ly` and independent of the length.

siteindex(x, y, Ly) = (x - 1) * Ly + mod1(y, Ly)

"""
    cylinder_bonds(Lx, Ly; periodic_y = true) -> Vector{Tuple{Int,Int}}

Nearest-neighbour bonds of an `Lx × Ly` lattice in column-major order, always with the smaller
linear index first.

For `Ly == 2` a periodic wrap would emit the single rung bond twice, doubling its coupling, so
the wrap is dropped and a notice is issued.
"""
function cylinder_bonds(Lx, Ly; periodic_y = true)
    wrap = periodic_y && Ly > 2
    if periodic_y && Ly == 2
        @info "cylinder_bonds: Ly == 2, dropping the periodic wrap (it would double-count the rung)"
    end
    bonds = Tuple{Int, Int}[]
    for x in 1:Lx, y in 1:Ly
        if y < Ly || wrap
            push!(bonds, minmax(siteindex(x, y, Ly), siteindex(x, y + 1, Ly)))
        end
        x < Lx && push!(bonds, minmax(siteindex(x, y, Ly), siteindex(x + 1, y, Ly)))
    end
    return unique!(bonds)
end

ladder_bonds(Lx, Ly) = cylinder_bonds(Lx, Ly; periodic_y = false)

# ## Reporting
#
# Two numbers matter. The *reduced* bond dimension counts the symmetry-resolved indices the MPO
# actually carries — this is what downstream DMRG pays for. The *dense-equivalent* bond dimension
# is the qdim-weighted sum of the per-sector multiplicities: what a symmetry-agnostic MPO would
# need for the same operator.

bonddim(secs, b) = length(secs[b])
# `init = 0`: an aggressive `SequentialSVD` truncation can leave a bond with no sectors at all.
densedim(secs, b) = sum(dim(c) for c in secs[b]; init = 0)
maxbonddim(secs) = maximum(b -> bonddim(secs, b), eachindex(secs))
maxdensedim(secs) = maximum(b -> densedim(secs, b), eachindex(secs))

"""
    build(name, H::OperatorSum; alg) -> NamedTuple

Construct the MPO and report its bond dimensions and (warm) construction time.
"""
function build(name, H::OperatorSum; alg = VertexCover(), quiet = false)
    mpo(H, alg)   # warm up compilation so the timing is meaningful
    stats = @timed mpo(H, alg)
    Ws, secs = stats.value.Ws, stats.value.bondsectors
    if !quiet
        println(
            rpad(name, 28),
            " N=", lpad(length(H.lattice), 4),
            "  nterms=", lpad(length(H), 6),
            "  D=", lpad(maxbonddim(secs), 4),
            "  D_dense=", lpad(maxdensedim(secs), 5),
            "  ", round(stats.time; digits = 4), " s"
        )
    end
    return (; Ws, secs, D = maxbonddim(secs), Ddense = maxdensedim(secs), time = stats.time)
end

# ## Verification
#
# Three checks of increasing cost; every example runs at least the first.
#
# **1. Faithfulness** — purely symbolic, valid at any `N` and for any sector including fermionic
# ones: the reduced MPO reconstructs the original term sum exactly
# ([`islossless`](@ref OpSum.islossless)). Only meaningful for lossless compression.
#
# **2. Tensor assembly** — contract the assembled `TensorMap`s with
# [`mpo_tensormap`](@ref OpSum.mpo_tensormap) and compare against the dense oracle. Exponential in
# `N`, so small systems only, but it stays inside TensorKit and is valid for fermions too.

function mpo_matches_oracle(H::OperatorSum; alg = VertexCover())
    m = mpo(H, alg)
    return mpo_tensormap(mpo_tensors(m)) ≈ instantiate(H)
end

# **3. Spectrum** — the physics check, and the only honest route for fermions when comparing
# against an external reference. Each block is labelled by a coupled sector `c` and holds one
# eigenvalue per multiplet; repeating it `dim(c)` times (the ``2j+1`` states of an SU(2) multiplet)
# makes spectra comparable across different symmetry groups.

# An operator with a trailing charge leg is reduced to an endomorphism first.
_endo(O::AbstractTensorMap) = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))

function spectrum(O::AbstractTensorMap)
    vals = Float64[]
    for (c, b) in blocks(_endo(O))
        ev = real(eigvals(Matrix(b)))
        for _ in 1:dim(c)
            append!(vals, ev)
        end
    end
    return sort!(vals)
end
spectrum(H::OperatorSum) = spectrum(instantiate(H))

"Relative deviation from Hermiticity — the sharpest detector of a wrong fermionic sign."
function hermiticity_error(H::OperatorSum)
    O = _endo(instantiate(H))
    return norm(O - O') / norm(O)
end
