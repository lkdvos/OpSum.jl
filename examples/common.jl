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
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms, mpo_tensormap, islossless, instantiate,
    Term, Terms, TermSum, opsum, lattice, spin, spin_ops, fermion_ops, couple, project,
    matrixunit, BipartiteAlgorithm, SVDBondAlgorithm
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, eigvals, norm

# ## Writing down operators
#
# `instances(IrrepOperator, V)` enumerates the on-site alphabet as `(charge, n)` pairs, where `n`
# indexes TensorKit's canonical block ordering. Hard-coding those `n`s is fragile: the ordering
# depends on how `V` was written down, so permuting the sectors of `V` silently permutes every
# letter index and you end up with a *different Hamiltonian and no error*.
#
# So the examples go the other way round: write the operator down as a `TensorMap` and let
# [`project`](@ref OpSum.project) expand it in the alphabet. `matrixunit(V, out, in)` is the shorthand for
# ``|out⟩⟨in|``. Both live in OpSum itself, so there is nothing to define here — see
# `src/operators/irrepprojection.jl`. Two properties matter for the pages that follow:
#
#  * the expansion is a `SiteOperator`, so ordinary arithmetic reads the way you would write it on
#    paper (`Sᶻ = (n↑ - n↓)/2`), and `A[i]` places the whole expansion on site `i`;
#  * `couple(A[i], B[j])` distributes over composite operands and drops letter pairs whose charges
#    cannot fuse to the total, so nothing has to be expanded by hand. The total defaults to the unit
#    sector — what a Hamiltonian term needs — and `to` names it otherwise. Under an abelian symmetry
#    the variadic `couple(A[i], B[j], C[k], …)` folds a whole chain, since every intermediate charge
#    is forced; non-abelian chains nest so each channel is named.
#
# Nothing is ever converted to a dense array, which matters for fermionic sectors where
# `convert(Array, t)` is not a well-defined operation.
#
# Under an abelian symmetry the operands of `couple` may be written in any site order — the stored
# form is site-ordered, so an out-of-order leg is inserted and the braiding phase (for fermions, the
# anticommutation sign) comes with it. Non-abelian coupling still needs increasing site indices, since
# reordering there would need F-moves. `H'` gives the hermitian-conjugate partner either way.
#
# Two builders bundle the conventions that would otherwise be re-derived per page:
# [`spin_ops`](@ref OpSum.spin_ops) for the U(1)-graded `(Sp, Sm, Sz)` and
# [`fermion_ops`](@ref OpSum.fermion_ops) for `(c, cd, n)`. Both, like `spin` and `matrixunit`, are
# memoised, so they can be called inside a term loop.

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
linear index first (`couple` only accepts left-to-right site order).

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
densedim(secs, b) = sum(dim(c) for c in secs[b])
maxbonddim(secs) = maximum(b -> bonddim(secs, b), eachindex(secs))
maxdensedim(secs) = maximum(b -> densedim(secs, b), eachindex(secs))

"""
    build(name, H; alg) -> NamedTuple

Construct the MPO and report its bond dimensions and (warm) construction time.
"""
function build(name, H::TermSum; alg = BipartiteAlgorithm(), quiet = false)
    sites = lattice(H)
    irrep_mpo(H, alg)   # warm up compilation so the timing is meaningful
    stats = @timed irrep_mpo(H, alg)
    Ws, secs = stats.value
    if !quiet
        println(
            rpad(name, 28),
            " N=", lpad(length(sites), 4),
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
# Three checks of increasing cost. Every example runs at least the first.
#
# **1. Faithfulness** — cheap, purely symbolic, valid at any `N` and for any sector including
# fermionic ones: the reduced MPO reconstructs the original term sum exactly. Only meaningful for
# lossless compression; after a truncating SVD it says nothing. This is
# [`islossless`](@ref OpSum.islossless), part of OpSum's public API.

# **2. Tensor assembly** — contract the assembled `TensorMap`s with
# [`mpo_tensormap`](@ref OpSum.mpo_tensormap) and compare against the dense oracle. Exponential in
# `N`, so small systems only, but it stays inside TensorKit and never builds a dense array, so it is
# valid for fermions too.

function mpo_matches_oracle(H::TermSum; alg = BipartiteAlgorithm())
    sites = lattice(H)
    Ws, secs = irrep_mpo(H, alg)
    return mpo_tensormap(irrep_mpo_tensors(Ws, secs, sites)) ≈ instantiate(H)
end

# **3. Spectrum** — the physics check, and the only honest route for fermions when comparing
# against an external reference. Eigenvalues are computed block by block, which is intrinsic and
# basis-independent.

# Each block is labelled by a coupled sector `c` and holds one eigenvalue per multiplet. In the
# full (dense) Hilbert space every such eigenvalue appears `dim(c)` times — for SU(2) that is the
# ``2j+1`` states of the multiplet. Repeating them accordingly is what makes spectra comparable
# across different symmetry groups.
function spectrum(O::AbstractTensorMap)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    vals = Float64[]
    for (c, b) in blocks(Oop)
        ev = real(eigvals(Matrix(b)))
        for _ in 1:dim(c)
            append!(vals, ev)
        end
    end
    return sort!(vals)
end
spectrum(H::TermSum) = spectrum(instantiate(H))

"Relative deviation from Hermiticity — the sharpest detector of a wrong fermionic sign."
function hermiticity_error(H::TermSum)
    O = instantiate(H)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    return norm(Oop - Oop') / norm(Oop)
end
