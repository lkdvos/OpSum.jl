# # Shared utilities
#
# Every example page begins by `include`ing this file. It contains no physics — only the
# scaffolding that the examples share:
#
#  1. a way to *name* the on-site alphabet letters instead of hard-coding indices,
#  2. a distributor that lets `couple` accept composite (multi-letter) local operators,
#  3. quasi-2D lattice geometry,
#  4. reporting of bond dimensions and construction times,
#  5. three tiers of verification.
#
# The file is plain Julia, so the same `include` works whether you are reading the rendered docs
# or running an example directly from a checkout.

using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms, instantiate, TermSum,
    spin, scalarop, couple, BipartiteAlgorithm, SVDBondAlgorithm
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, eigvals, norm

# `LocalOp` is not exported; this shorthand promotes a bare alphabet letter to a local operator.
LO(x) = OpSum.LocalOp(x)

# ## Naming the alphabet letters
#
# `instances(IrrepOperator, V)` enumerates the on-site alphabet as `(charge, n)` pairs, where `n`
# indexes TensorKit's canonical block ordering. Hard-coding those `n`s is fragile: the ordering
# depends on how `V` was written down, so permuting the sectors of `V` silently permutes every
# letter index and you end up with a *different Hamiltonian and no error*.
#
# So we go the other way round. We write down the operator we want as a `TensorMap`, then expand
# it in the alphabet. The alphabet is orthonormal under TensorKit's qdim-weighted inner product,
# so the expansion coefficients are just inner products. Crucially this never converts anything
# to a dense array, which matters for fermionic sectors where `convert(Array, t)` is not a
# well-defined operation.
#
# An expanded operator is a small wrapper around a list of `letter => coefficient` pairs, so that
# ordinary operator arithmetic (`Sᶻ = (n↑ - n↓)/2`) reads the way you would write it on paper.

struct SiteOp{I <: Sector}
    terms::Vector{Pair{IrrepOperator{I}, ComplexF64}}
end

function Base.:*(x::Number, a::SiteOp{I}) where {I}
    return SiteOp{I}(Pair{IrrepOperator{I}, ComplexF64}[l => ComplexF64(x) * v for (l, v) in a.terms])
end
Base.:*(a::SiteOp, x::Number) = x * a
Base.:/(a::SiteOp, x::Number) = inv(x) * a
Base.:-(a::SiteOp) = -1 * a
Base.:+(a::SiteOp{I}, b::SiteOp{I}) where {I} = SiteOp{I}(_combine(I, a.terms, b.terms))
Base.:-(a::SiteOp{I}, b::SiteOp{I}) where {I} = a + (-b)

function _combine(::Type{I}, a, b; atol = 1.0e-12) where {I}
    acc = Dict{IrrepOperator{I}, ComplexF64}()
    for (l, v) in Iterators.flatten((a, b))
        acc[l] = get(acc, l, zero(ComplexF64)) + v
    end
    return Pair{IrrepOperator{I}, ComplexF64}[l => v for (l, v) in acc if abs(v) > atol]
end

"""
    ito_expand(O::AbstractTensorMap, V) -> SiteOp

Expand the single-site operator `O : V ← V ⊗ Vect[I](c => 1)` in the ITO alphabet of `V`.
Throws if the result does not reproduce `O`, so a change in TensorKit's conventions surfaces
here rather than as wrong physics downstream.
"""
function ito_expand(O::AbstractTensorMap, V::ElementarySpace; atol = 1.0e-10)
    I = sectortype(V)
    c = only(sectors(domain(O)[2]))
    letters = [l for l in instances(IrrepOperator, V) if l.c == c]
    coeffs = [l => ComplexF64(dot(instantiate(l, V), O)) for l in letters]
    filter!(p -> abs(last(p)) > atol, coeffs)
    isempty(coeffs) && throw(ArgumentError("operator is zero on the alphabet of $V"))
    err = norm(sum(v * instantiate(l, V) for (l, v) in coeffs) - O)
    err < atol * max(1, norm(O)) ||
        throw(ArgumentError("alphabet expansion failed (residual $err)"))
    return SiteOp(coeffs)
end

"""
    matrixunit(V, out, in) -> SiteOp

The matrix unit ``|out⟩⟨in|`` on a space whose sectors are one-dimensional and multiplicity-one
(abelian or fermionic: `Vect[U₁]`, `Vect[FermionNumber]`, products thereof). The operator charge
follows from fusion, `out ⊗ in' → c`, and the letter index is *derived* rather than guessed.
"""
function matrixunit(V::ElementarySpace, out::I, in::I) where {I <: Sector}
    dim(out) == 1 && dim(in) == 1 ||
        throw(ArgumentError("matrixunit requires one-dimensional sectors"))
    dim(V, out) == 1 && dim(V, in) == 1 ||
        throw(ArgumentError("matrixunit requires multiplicity-one sectors in V"))
    c = only(⊗(out, dual(in)))
    t = zeros(ComplexF64, V ← V ⊗ Vect[I](c => 1))
    block(t, out) .= 1
    return ito_expand(t, V)
end

# ## Placing operators on the lattice
#
# A one-site term is a plain sum over the expansion. A two-site term has to distribute the
# coupling over both expansions, because `couple` insists on single-letter operands: it builds
# the caterpillar fusion tree one leg at a time, so `couple(Sᶻ[i], Sᶻ[j])` — with `Sᶻ` a
# two-letter operator — throws `"first operand must be a single-term operator"`.

"One-site term: place an expanded operator on site `i`."
onsite(A::SiteOp, i::Int) = sum(v * LO(l)[i] for (l, v) in A.terms)

"""
    bondterm(A, i, B, j; to) -> TermSum

Two-site term `A_i B_j` fused to total charge `to`, distributed over both alphabet expansions.
Letter pairs whose charges cannot fuse to `to` are dropped.

`couple` is strictly left-to-right, so `i < j` is required. For fermions this is not a mere
convention: swapping the two sites flips the sign of the term.
"""
function bondterm(A::SiteOp, i::Int, B::SiteOp, j::Int; to)
    i < j || throw(ArgumentError("bondterm requires i < j (`couple` is left-to-right); got $i, $j"))
    terms = [
        (va * vb) * couple(LO(la)[i], LO(lb)[j]; to = to)
            for (la, va) in A.terms, (lb, vb) in B.terms
            if !iszero(Nsymbol(la.c, lb.c, to))
    ]
    isempty(terms) && throw(ArgumentError("no letter pair of the operands fuses to $to"))
    return sum(terms)
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
    build(name, H, sites; alg) -> NamedTuple

Construct the MPO and report its bond dimensions and (warm) construction time.
"""
function build(name, H::TermSum, sites; alg = BipartiteAlgorithm(), quiet = false)
    irrep_mpo(H, sites, alg)   # warm up compilation so the timing is meaningful
    stats = @timed irrep_mpo(H, sites, alg)
    Ws, secs = stats.value
    if !quiet
        println(
            rpad(name, 28),
            " N=", lpad(length(sites), 4),
            "  nterms=", lpad(length(H.terms), 6),
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
# lossless compression; after a truncating SVD it says nothing.

function islossless(H::TermSum, sites; alg = BipartiteAlgorithm())
    Ws, secs = irrep_mpo(H, sites, alg)
    back = mpo_terms(Ws, secs)
    Set(keys(back.terms)) == Set(keys(H.terms)) || return false
    return all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
end

# **2. Tensor assembly** — contract the assembled `TensorMap`s and compare against the dense
# oracle. Exponential in `N`, so small systems only, but it stays inside TensorKit and never
# builds a dense array, so it is valid for fermions too.

"Contract a vector of MPO tensors into a single operator in `instantiate`'s convention."
function mpo_tensormap(Ts)
    N = length(Ts)
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = ncon(Ts, net)                # legs: o₁…o_N, i₁…i_N, bL, bR
    O = removeunit(O, 2N + 1)        # drop the trivial left boundary
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end

function mpo_matches_oracle(H::TermSum, sites; alg = BipartiteAlgorithm())
    Ws, secs = irrep_mpo(H, sites, alg)
    return mpo_tensormap(irrep_mpo_tensors(Ws, secs, sites)) ≈ instantiate(H, sites)
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
spectrum(H::TermSum, sites) = spectrum(instantiate(H, sites))

"Relative deviation from Hermiticity — the sharpest detector of a wrong fermionic sign."
function hermiticity_error(H::TermSum, sites)
    O = instantiate(H, sites)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    return norm(Oop - Oop') / norm(Oop)
end
